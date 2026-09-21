# StockMixer RL / PPO 改造计划

## 结论

PPO 可以从当前 SFT checkpoint 继续训练，但不能把现有确定性收益回归器直接套进 PPO。
必须增加随机策略、价值函数、按时间顺序运行的交易环境和 rollout/GAE/PPO 更新逻辑。
给其他队友实现的主方案是 **SFT 初始化、随后纯 PPO**：允许加载当前 SFT 权重作为初始策略，但
PPO 阶段不使用 MSE、监督 Rank loss、行为克隆或对 SFT 的 KL 辅助项。完整推荐参数已经固化在
`configs/stockmixer_ppo_recommended.json`。

## 环境定义

每个 episode 必须严格按交易日推进：

- 状态 `s_t`：截至 `t` 的因果归一化窗口、当日 eligibility/涨跌停信息、上一日 Top10% 持仓。
- 动作 `a_t`：4,650只股票的随机连续 score；训练时从逐股票高斯策略采样，验证时使用均值。
  当日 Top10% 持仓和完整横截面排序都由这条 sampled score 决定。
- 转移：推进到下一交易日，不能随机打乱相邻日顺序。
- 日奖励：

```text
r_t = 0.4 * RankIC_t
    + 0.3 * 252 * (Top10Return_t - MarketReturn_t)
    + 0.3 * (1 - JaccardTurnover_t)
```

按日取均值后与官方总分同构。实现时应减去与动作无关的常数基线并标准化 advantage，但日志必须
始终报告未经改写的三个官方分量和精确总分。Rank IC 使用所有有标签股票；收益与换手率使用排除
涨停股票后的可交易集合，必须与 `evaluation/evaluate.py` 一致。

训练 episode 只使用训练区间，验证 episode 使用固定验证区间。测试集在 PPO 设计与超参数完全冻结
前不可访问。

## 模型改动

1. 把 `StockMixerReturn` 拆为可复用 backbone 和每股票表示/分数输出。
2. 从 `artifacts/stockmixer_sft_current/best/model.pt` 加载 SFT backbone。
3. 增加 policy head，输出 4,650 只股票的 logits，并对不可交易股票施加 `-inf` mask。
4. 增加 previous-holding 条件。最低成本版本可在 score mean 上加入可学习的持仓偏置；更完整版本使用
   小 MLP 融合 `base_score`、上一日持仓与 eligibility。没有该状态，策略无法主动控制换手率。
5. 增加 value head 输出 `V(s_t)`；可以共享 backbone，但 policy/value head 分开。
6. policy head 输出逐股票高斯均值，并维护有界的 `log_std`。均值先做当日横截面 z-score，训练时
   采样每只股票的连续 score 并保存逐股票 old log-prob；验证时直接使用均值排序。

这里不能只把随机 Top-K 当作动作、同时用确定性 logits 计算 Rank IC：在 likelihood-ratio 梯度中，
这个 IC 项对已采样 Top-K 动作是常数，写进 reward 也不会提供正确的 Rank IC 学习信号。连续 score
动作让完整 Rank IC、Top10%收益和换手率都真正依赖 sampled action。逐股票策略近似按共享全局
advantage 做 MAPPO-style PPO clipping；4,650维联合 log-prob 只做监控，不直接形成一个 joint ratio，
否则 ratio 几乎必然饱和。该近似必须作为方法假设写入实验报告。

Plackett-Luce/Gumbel-TopK 可以作为第二个消融，但此时应把 Rank IC 权重设为0；若仍要用完整综合分，
必须采样完整排序并解决高维 joint-ratio 稳定性，不能把确定性尾部排序冒充随机动作。

## 推荐的纯 PPO 配置

基线设置：

| 参数 | 建议值 |
|---|---:|
| SFT 初始化 | 当前 `best/model.pt` |
| lookback | 64 |
| 训练动作 | 逐股票高斯连续 score，随后精确排序/取 Top10% |
| score mean | 当日横截面 z-score |
| 初始 `log_std` / 范围 | `-1.5` / `[-4, 0]` |
| PPO ratio | 逐股票 MAPPO-style ratio，禁止4,650维 joint ratio |
| 验证动作 | score mean 的确定性排序 |
| 16卡每卡 rollout | 32 个连续交易日 |
| PPO epochs / rollout | 4 |
| minibatches / PPO epoch | 4 |
| `gamma` | 1.0 |
| GAE lambda | 0.95 |
| policy clip | 0.1 |
| value clip | 0.2 |
| policy LR | `3e-5` |
| value LR | `1e-4` |
| entropy coefficient | `1e-3` |
| target KL | 0.01 |
| gradient clip | 0.5 |
| 最大 policy updates | 100 |

`gamma=1.0` 是因为目标是有限历史区间内各日官方分数的平均，而不是偏好近期收益。三个分量分别按
各自有效交易日求均值后再加权，避免某一分量因缺失日数量不同而被隐式重权。日超额收益乘252会有
较大方差，使用 critic/control variate 和全局 advantage 标准化降方差，但第一版不 clip reward，避免
改变优化目标。advantage 必须在全部 DDP rank 汇总后统一标准化。每5个 policy updates 在固定验证
episode 上跑一次确定性策略；按最近3次验证精确总分的移动平均保存 checkpoint，连续6次验证不改善
才停止。

纯 RL 开关必须保持：

```text
supervised_mse_weight = 0
supervised_rank_weight = 0
behavior_cloning_weight = 0
kl_to_sft_weight = 0
```

PPO 新旧策略之间用于 clip/target-KL 的约束仍然必须存在，它是 PPO 算法的一部分，不属于 SFT 辅助。
value loss 同样属于 actor-critic，不属于监督收益回归。

## 16卡 rollout 约束

- 每个 rank 获取不同的连续日期片段，片段内部不得 shuffle。
- 上一日持仓必须随环境状态传递；片段首日从空仓或明确保存的前序持仓开始。
- 不允许16个 rank 重复采样同一日期轨迹后仅改变随机 seed，这会虚增样本量。
- rollout buffer 建议只存日期索引、action、old log-prob、value、reward 和 mask；训练时从 memmap
  重建窗口，避免缓存 `4650×64×6` 状态导致内存膨胀。
- 每次 PPO update 前同步当前 policy；完成 rollout 后汇总 advantage 统计量，再进行 DDP 更新。
- 若 observed KL 超过 `target_kl=0.01`，提前结束当前 rollout 的 PPO epochs。
- 同时记录逐股票 approximate KL、仅供诊断的联合 KL、clip fraction、`log_std` 和 action entropy；
  若首轮更新 clip fraction 已接近1，应先降低 policy LR/增大 rollout，而不是继续训练。

## PPO 训练模块

需要新增：

- `finmodel/rl/environment.py`：因果交易环境和精确日奖励。
- `finmodel/rl/policy.py`：SFT backbone、policy head、value head、masked Top-K 分布。
- `finmodel/rl/buffer.py`：状态索引、动作、旧 log-prob、value、reward、done、mask。
- `finmodel/rl/ppo.py`：GAE、clipped surrogate、value loss、entropy、KL early stop。
- `scripts/train_stockmixer_ppo.py` 与独立 `configs/stockmixer_ppo.json`。
- 环境、奖励逐项对齐、动作 mask、checkpoint 重载与固定 seed 测试。

正式长跑前必须通过两个额外测试：改变 sampled score 应能改变 Rank IC reward；固定 sampled action
而只改变未进入动作的诊断 logits，不应改变 PPO reward。这样可以排除“IC 被记录但没有策略梯度”
这一类静默错误。

金融历史只有一条主要市场轨迹，PPO 很容易记忆日期，因此必须使用多个不重叠连续训练片段、固定
验证 episode 和多 seed。第一轮先用 seed 2026 验证机制；确认 reward、KL 和 entropy 曲线正常后，
再用 2027/2028 报告均值和标准差。

## PPU 环境

不需要重新创建 Python 环境。当前 PPU Torch、DDP/PCCL、SwanLab 和数据栈已经足够；PPO 可以直接
用 PyTorch 实现。暂不安装 Stable-Baselines3、Gymnasium 或 RLlib：它们对这个全市场、动态 mask、
DDP 的自定义动作空间帮助有限，还可能引入依赖冲突。

建议只新建独立的配置、SwanLab project 与 artifact 目录，继续使用：

```text
/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python
```

正式训练前仍需在 4 卡上做 policy sampling、log-prob、GAE、PPO backward、DDP 同步和 checkpoint
重载 smoke。若后续确实采用外部 RL 框架，再建立项目私有环境，并保持共享 PPU Torch 三件套不变。
