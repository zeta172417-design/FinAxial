# FinAxial D0

FinAxial D0 由两个模块组成：**预测器**（Predictor）和**决策器**（Decision-maker）。预测器用因果轴向 Transformer 编码历史行情和同日股票截面；决策器在每日预测分数之上学习保留持仓或换股。仓库提供两条决策器训练路线：同日组相对优势的 GRPO，以及同日组内标准化 GAE 优势的 PPO。

```text
历史 OHLCVA + 股票 ID
        ↓
双头预测器：排序分数 + 次日绝对收益预测
        ↓  排序分数、hidden、持仓状态
三动作 hysteresis 决策器
        ↓
每日完整排序与 Top 10% 持仓
```

当前决策器的 `return_feature_mode=proxy`：它读取排序分数及预测器 hidden，**尚未直接读取独立收益头的数值**。收益头作为共享编码器的辅助训练任务；把它接入决策器应作为新的对照实验，不能冒充现有最佳结果。完整结构和实验口径见 [D0 说明](docs/d0.md)。

## 当前结果

预测器从头训练，排序头优化 soft Rank IC，独立收益头优化按 `0.02` 缩放的绝对收益 Huber；两个 loss 共享编码器梯度。决策器均从 seed 2026 随机初始化，训练时冻结预测器。

| 路线 | 选中 epoch | Final | Rank IC | 年化超额 | `1−换手率` |
|---|---:|---:|---:|---:|---:|
| GRPO / 同日组优势 | 6 | **0.354955** | **0.071254** | **0.112191** | **0.975988** |
| PPO / GAE 同日组优势 | 5 | 0.346859 | 0.069086 | 0.104950 | 0.959132 |

这是 2025-01-02 至 2026-06-05 的 343 个有标签日期上的内部回放。该时段也用于模型选择，**不是独立留出测试集**。两条路线虽然每块都是 16 个 rollout，但分别在 16 卡与 4 卡上运行，随机流不逐 bit 相同。GRPO 是当前默认决策器权重；PPO 保留为研究路线。单卡重载复算结果见 [GRPO](artifacts/d0/evaluation_grpo.json) 和 [PPO](artifacts/d0/evaluation_ppo_group.json)。

## 模型与权重

| 模块 | 权重 | 参数量 |
|---|---|---:|
| 双头预测器 | [model.pt](artifacts/d0/predictor/best/model.pt) | 964,448 |
| 默认 GRPO 决策器 | [policy.pt](artifacts/d0/decision_grpo/best/policy.pt) | 7,798 |
| 对照 PPO 决策器 | [policy.pt](artifacts/d0/decision_ppo_group/best/policy.pt) | 7,798 |

原始 CSV、重构标签和约 4.5 GB 的预测器特征缓存不在仓库中；对应的处理代码、配置、模型权重和结果摘要在仓库中。源数据须由使用者自行合法获取。

## 数据准备与复现

本地放置行情 CSV 和对应有标签评测 CSV 后，构建只读面板：

```bash
PYTHON=${FINAXIAL_PYTHON:-python}
$PYTHON scripts/preprocess.py --source data/raw/训练集.csv --output artifacts/panel/train
$PYTHON scripts/build_phase1_panel.py \
  --train-panel artifacts/panel/train \
  --test-features evaluation/测试集_X.csv \
  --test-labels evaluation/测试集_Y.csv \
  --output artifacts/panel/phase1
```

规范配置是 [预测器](configs/d0_predictor.json)、[GRPO 决策器](configs/d0_decision_grpo.json) 和 [PPO 决策器](configs/d0_decision_ppo_group.json)。定制 PPU 环境中不要安装或覆盖厂商适配的 Torch。训练入口会自动加载 PPU SDK，并跳过已有完整结果：

```bash
bash scripts/run_d0_predictor_16ppu.sh
bash scripts/run_d0_decision_grpo_16ppu.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_d0_decision_ppo_4ppu.sh
```

在已构建缓存的机器上复算最佳 checkpoint：

```bash
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/evaluate_d0.py --route grpo
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/evaluate_d0.py --route ppo_group
```

运行测试：`$PYTHON -m unittest discover -s tests`。先前的实验启动文件和报告已移到本地忽略的归档目录 `artifacts/archive_d0_20260925/`；Git 历史中也可追溯旧公开版本。

## License

原创代码采用 [MIT License](LICENSE)。外部数据不随仓库分发。
