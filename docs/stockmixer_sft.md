# StockMixer final-only SFT

## 最终配置

当前唯一主配置是 `configs/stockmixer_sft.json`：

- lookback：64个交易日。
- 训练：20 epochs、seed 2026、16卡 DDP。
- 输入：每只股票、每个窗口、每个特征分别做因果 z-score，裁剪到 `[-5,5]`。
- 优化器：AdamW，LR `1e-3`，weight decay `1e-5`。
- 调度：前2 epochs线性 warmup，随后 cosine 到初始 LR 的1%。
- EMA：0.99。
- loss：只有比赛综合分的可微近似，`mse_weight=0`。
- checkpoint：按 EMA 精确验证总分最近5次的移动平均选择。

训练代理：

```text
soft_final = 0.4 * soft_rank_ic
           + 0.3 * [0.5 * tanh(raw_annual_excess / 0.5)]
           + 0.3 * soft_top10_jaccard
loss = -soft_final
```

`soft_rank_ic` 是预测 soft percentile rank 与真实收益 hard percentile rank 的 Pearson 相关；
soft Top10% 使用 sigmoid membership；相邻日 fuzzy Jaccard 对齐官方 `1-turnover`。Rank IC 保留所有
有标签股票，Top10%收益和换手率排除涨停股票，与精确 evaluator 的两个 universe 一致。

## 数据切分

```text
训练：2018-01-02 至 2024-07-04
边界排除：2024-07-05
验证：2024-07-08 至 2024-12-31（120日）
```

边界日被排除是因为其标签引用下一段价格。验证集只用于选择 checkpoint，不参与梯度更新。

## 正式结果

20-epoch、16卡 run 完整结束，epoch 20 同时是稳定 EMA 选择点：

| 项目 | 数值 |
|---|---:|
| exact validation final score | **0.3239581266** |
| 5-epoch trailing mean | 0.3227935766 |
| 5-epoch std | 0.0010693435 |
| Rank IC | 0.0471290957 |
| 年化超额收益 | 0.2544839204 |
| `1-turnover` | 0.7625377075 |
| optimizer updates | 1,960 |
| 单卡峰值显存 | 1,283,302,912 bytes |

raw epoch 20 的单点总分为 `0.3241667846`，略高于 EMA，但主 checkpoint 遵守预先固定的 EMA/稳定
选择规则，不使用事后单点挑选。由于不含 MSE，预测绝对幅度可漂移，MSE/MAE 仅供诊断；比赛分数
由排序和持仓集合决定。

## 16卡启动

```bash
source /usr/local/PPU_SDK/envsetup.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python \
  scripts/run_stockmixer_sft_sweep.py \
  --config configs/stockmixer_sft.json \
  --panel artifacts/panel/train \
  --nproc-per-node 16 \
  --lookbacks 64 \
  --root artifacts/stockmixer_sft_runs
```

SwanLab project 为 `financial-modeling-stockmixer-sft-final-only-ep20`。rank 0 独占日志写入，其他
rank 不创建重复 run。训练脚本拒绝超过配置中20 epochs的命令行覆盖。

## 100-epoch 对照为何废弃

同配置延长到100 epochs后，验证总分在早期达到峰值，随后持续回落；停止时已到 epoch 66，EMA
总分降到约0.24。因此100-epoch运行被标记为过拟合并从主产物中删除，不作为报告或初始化权重。
