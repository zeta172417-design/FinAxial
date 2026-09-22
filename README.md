# FinAxial C0

FinAxial 是面向大规模股票截面的因果轴向 Transformer。C0 同时建模单只股票的时间依赖与同一日期的跨股票关系，并直接优化由排序能力、Top 10% 超额收益和组合稳定性组成的可微目标。

本仓库提供模型实现、三组可复现权重、因果数据处理、训练脚本、参考测评器和完整测试结果。原始行情数据、重构标签、memmap 面板与在线实验缓存不随仓库发布。

## 架构

C0 每个样本包含 4,650 只股票、32 个 burn-in 日期和 64 个并行监督日期：

```text
OHLCVA [stocks, 96, 6]
  -> 每个 token 的 trailing-64 因果 z-score
  -> 特征投影 + 门控公司 embedding
  -> Temporal Attention 1（RoPE、sliding causal mask）
  -> Stock Attention 1（同日全截面、无股票位置编码）
  -> Temporal Attention 2
  -> 最后 64 日共享 Stock Attention 2
  -> LayerNorm + 线性头
  -> [64, 4650] 截面排序信号
```

主要规格：

- `d_model=128`，4 heads，FFN 512；
- 2 个 Temporal block 与 2 个 Stock block，交错轴向排列；
- 公司 embedding 为 32 维，通过学习门控加入行情 token；
- 共 964,064 个参数；
- 时间注意力使用 RoPE，股票注意力不使用位置编码；
- 所有输出严格因果，日期 `t` 的预测看不到 `t` 之后的输入。

训练目标为：

```text
soft_score = 0.4 * soft_rank_ic
           + 0.3 * bounded_soft_top10_annual_excess
           + 0.3 * soft_portfolio_stability
loss = -soft_score
```

Rank 与 Top 10% 的 soft temperature 在前 20 epochs 余弦退火。训练使用 AdamW、`3e-4` 学习率、5 epochs warmup、cosine decay、weight decay `0.05`，最多 30 epochs。

## 测试结果

完整有标签测试期为 2025-01-02 至 2026-06-05，共 343 个交易日。前三组为单 seed，最后一组对每日截面 percentile rank 做等权平均。

| 推理口径 | Raw Final | Rank IC | 年化超额收益 | `1 - turnover` |
|---|---:|---:|---:|---:|
| seed 2026 | 0.335148 | 0.058924 | 0.169437 | 0.869158 |
| seed 2027 | 0.316601 | 0.055210 | 0.145721 | 0.836004 |
| seed 2028 | **0.335275** | **0.060288** | **0.170325** | 0.866874 |
| 三 seed 等权 rank 集成 | 0.330705 | 0.058997 | 0.166766 | 0.856920 |

单 seed Raw Final 均值为 `0.329008 ± 0.010745`。固定的因果 `EWMA(alpha=0.25)` 将三 seed 集成 Final 提高至 `0.337911`，对应 Rank IC `0.048561`、年化超额收益 `0.162433`、`1-turnover` `0.899189`。

其中 2025 年区间曾用于 checkpoint 与架构开发。完全独立、此前锁定的 2026-01-05 至 2026-06-05 共 100 日结果如下：

| 口径 | Raw Final | Rank IC | 年化超额收益 | `1 - turnover` |
|---|---:|---:|---:|---:|
| 三 seed 均值 | 0.288069 ± 0.007601 | 0.037078 | 0.055887 | 0.854906 |
| 三 seed 等权 rank 集成 | **0.288593** | **0.037613** | **0.059014** | 0.852813 |
| 集成 + 固定 EWMA 0.25 | **0.294852** | 0.030040 | 0.043926 | **0.898861** |

完整机器可读结果位于 [evaluation_results.json](artifacts/finaxial_c0/evaluation/evaluation_results.json)。由于 343 日汇总包含开发区间，它用于完整回放；100 日结果更适合衡量独立泛化能力。

## 权重

| Seed | Checkpoint | SHA-256 |
|---:|---|---|
| 2026 | `artifacts/finaxial_c0/seed_2026/model.pt` | `3b186d87c1226ab793c755205f698a0974353441320f00d3e9c4cd812989b9c1` |
| 2027 | `artifacts/finaxial_c0/seed_2027/model.pt` | `5084c4ec4bdbd09a2d0e29f7d9c68ffc8da3ddf4234faa4ae2320b1973219e92` |
| 2028 | `artifacts/finaxial_c0/seed_2028/model.pt` | `88b634b73fa102ebca85a1918c5d2212d0bf233f8daffcabe02da2dd2262e6b2` |

推荐使用三 seed rank 集成来降低初始化方差；资源受限时可固定使用任一预先选定的 seed，不应根据目标测试区间事后选择 seed。

## 数据准备

请在本地准备：

```text
data/raw/训练集.csv
data/raw/测试集_X.csv
evaluation/测试集_X.csv
evaluation/测试集_Y.csv
```

构建只读面板：

```bash
PYTHON=${FINAXIAL_PYTHON:-python}

$PYTHON scripts/preprocess.py \
  --source data/raw/训练集.csv \
  --output artifacts/panel/train

$PYTHON scripts/build_phase1_panel.py \
  --train-panel artifacts/panel/train \
  --test-features evaluation/测试集_X.csv \
  --test-labels evaluation/测试集_Y.csv \
  --output artifacts/panel/phase1
```

缺失 OHLC 使用此前最近收盘价因果填充，成交量与成交额置零；上市前数据保持无效。标签定义为 `close(t+1) / close(t) - 1`。任何窗口归一化都只使用截至当前 token 的历史。

## 训练

规范配置为 [configs/c0.json](configs/c0.json)。8 张 PPU 的复现入口会先运行单卡 smoke：

```bash
bash scripts/run_c0_8ppu.sh
```

复现其他 seed 时复制配置并只修改 `seed`、SwanLab run 名及输出目录。可通过 `FINAXIAL_PYTHON` 指定解释器；在定制加速器环境中不要擅自替换厂商适配的 Torch 套件。

## 测评

在完整有标签测试期复现三 seed 与集成结果：

```bash
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/evaluate_c0.py \
  --manifest configs/c0_evaluation.json \
  --panel artifacts/panel/phase1 \
  --output artifacts/finaxial_c0/evaluation \
  --device cuda:0
```

为防止误覆盖，已有结果存在时脚本会拒绝重复执行；只有明确审计时才使用 `--allow-rerun`。

## 测试

```bash
$PYTHON -m unittest discover -s tests -v
```

测试覆盖因果填充、未来数据隔离、轴向注意力因果性、股票重排等变性、soft-score 反向传播、checkpoint 重载与指标复算。

## License

原创代码使用 [MIT License](LICENSE)。外部数据不随仓库分发，使用者需自行确认其数据许可。
