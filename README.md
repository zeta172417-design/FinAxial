# Stock-Time Transformer B0

本仓库保留赛题五实验最终选定的唯一模型：Stock-Time Transformer B0。旧 StockMixer、Kronos、
V1/V2-A/K16 和 B1 实现及权重均已移除。

## 模型

B0 接收完整股票截面，使用32个burn-in日期和64个连续监督日期：

```text
OHLCVA [stocks, 96, 6]
  -> 每个token的trailing-64因果z-score
  -> 2层sliding causal Temporal Attention + RoPE
  -> 最后64个日期
  -> 每日共享的2层Stock Attention（股票轴无位置编码）
  -> 64个日截面收益排序信号
```

训练目标直接近似比赛综合分：

```text
soft_final = 0.4 * mean(daily_soft_rank_ic)
           + 0.3 * bounded(mean(daily_annual_excess))
           + 0.3 * mean(63 adjacent soft_jaccard)
loss = -soft_final
```

模型共609,728个参数。参数EMA已取消，checkpoint按raw Final连续3轮均值选择；信号端固定额外报告
`alpha=0.25`的因果EWMA。详细结构和指标见[模型文档](docs/model.md)。

## 已保存结果

完整343日事后重构验证集上的选定checkpoint：

| 指标 | Raw | 信号EWMA 0.25 |
|---|---:|---:|
| Final score | **0.328791** | **0.336329** |
| Rank IC | 0.054986 | 0.045317 |
| 年化超额收益 | 0.158661 | 0.152748 |
| `1 - turnover` | 0.863996 | 0.907927 |

checkpoint 位于`artifacts/stock_time_transformer_b0/best/model.pt`，SHA256 为
`5076c4560ccc0b2f7d4238515423eaa145ef2e335e58a24c06553f5d0980c891`。

## 数据

GitHub 不包含比赛 CSV、重构标签、memmap 面板或 SwanLab 缓存。请准备：

```text
data/raw/训练集.csv
data/raw/测试集_X.csv
evaluation/测试集_X.csv
evaluation/测试集_Y.csv
```

原始文件的大小和校验信息见`data/raw/README.md`。构建面板：

```bash
PYTHON=/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python
$PYTHON scripts/preprocess.py \
  --source data/raw/训练集.csv \
  --output artifacts/panel/train

$PYTHON scripts/build_phase1_panel.py \
  --train-panel artifacts/panel/train \
  --test-features evaluation/测试集_X.csv \
  --test-labels evaluation/测试集_Y.csv \
  --output artifacts/panel/phase1
```

## 复现训练

正式结果使用8张PPU、30 epochs。启动脚本会先做单卡smoke：

```bash
bash scripts/run_stock_time_transformer_8ppu.sh
```

核心配置为`configs/stock_time_transformer.json`，固定解释器为
`/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python`。不要升级或替换共享环境中的Torch套件。

## 历史窗口评测

同一批217个交易日上的位置对照表明，预测能力从32日历史增长到约64日，之后基本饱和。复现：

```bash
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.run \
  --master-port=29740 --nproc-per-node=2 \
  scripts/evaluate_stock_time_transformer_context.py \
  --config configs/stock_time_transformer.json \
  --checkpoint artifacts/stock_time_transformer_b0/best/model.pt \
  --panel artifacts/panel/phase1 \
  --output artifacts/context_window_eval
```

完整分析见[历史窗口报告](docs/context_window_evaluation.md)。

## 测试

```bash
$PYTHON -m unittest discover -s tests -v
```

测试覆盖因果填充、未来隔离、模型因果性、股票重排等变性、soft-final反向传播、checkpoint重载
以及本地指标与官方`evaluate.py`逐项一致。

## 许可

原创代码使用MIT License。比赛数据和官方题目文件不随仓库分发。
