# StockMixer Soft-Final for Cross-Sectional Stock Ranking

这是赛题五的可复现 StockMixer 实验仓库。当前唯一主版本使用64日窗口、20 epochs、16卡 DDP，
直接优化比赛综合分的可微近似，不加入 MSE：

```text
soft_final = 0.4 * soft_rank_ic
           + 0.3 * bounded_soft_annual_excess
           + 0.3 * soft_top10_jaccard
loss = -soft_final
```

## 当前结果

固定验证集为 2024-07-08 至 2024-12-31，共120个交易日。当前选定 checkpoint 是 EMA epoch 20：

| 指标 | 验证结果 |
|---|---:|
| Final score | **0.323958** |
| Rank IC | 0.047129 |
| 年化超额收益 | 0.254484 |
| `1 - turnover` | 0.762538 |
| 覆盖率 | 0.989839 |

checkpoint 位于 `artifacts/stockmixer_sft_current/best/model.pt`，SHA256 为
`10692c8145083699675c7084379d322d569862ba26e0764ad0d2c8bb2c2d854c`。MSE/MAE 只作为诊断日志，
不参与训练。完整配置和实验说明见 [SFT 文档](docs/stockmixer_sft.md)。

## 数据不随仓库分发

GitHub 仓库不包含比赛原始 CSV、事后重构测试标签、memmap 面板、SwanLab 缓存或下载的模型权重。
请从比赛官方渠道取得训练集，并放置为：

```text
data/raw/训练集.csv
```

期望字段：

```text
ts_code,trade_date,open,high,low,close,vol,amount,
y_ret_1d,flag_limit_up,flag_limit_down
```

本次使用文件的大小、日期范围和 SHA256 见 [数据清单](data/raw/README.md)。代码不会修改原始 CSV。

## 获取代码

StockMixer 以固定 commit 的 Git submodule 引用。本项目对上游的少量兼容修改以 patch 保存：

```bash
git clone --recurse-submodules <repository-url>
cd financial_modeling
bash scripts/setup_third_party.sh
```

如果克隆时没有加 `--recurse-submodules`，setup 脚本会自动初始化它们。

## 环境

不要在共享 PPU 环境中升级或替换 `torch`、`torchvision`、`torchaudio`。本次实验使用：

```text
/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python
torch 2.11.0 PPU build
swanlab 0.9.8
```

普通 Python 环境可安装项目的纯 Python 依赖；Torch 应按目标硬件单独安装：

```bash
python -m pip install -e .
```

## 构建因果面板

```bash
PYTHON=/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python
$PYTHON scripts/preprocess.py \
  --source data/raw/训练集.csv \
  --output artifacts/panel/train
```

处理规则：OHLC 缺失只用此前最近收盘价填充，vol/amount 缺失置零，上市前保持无效；所有窗口
归一化只使用截至预测日的当前窗口。生成的 manifest 会记录源文件哈希、形状、日期范围和处理版本。

## 16卡复现实验

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

运行参数固定为 AdamW、LR `1e-3`、weight decay `1e-5`、2-epoch linear warmup、cosine decay、
gradient clip 1.0、EMA 0.99。每个 rank 每次处理一个相邻交易日对，全局 batch 为16个日期对。

## 测试

```bash
$PYTHON -m unittest discover -s tests -v
```

测试覆盖因果填充、未来数据隔离、官方指标对齐、soft-final loss、PPU兼容模型算子、checkpoint 重载
和固定 seed。测试集数据不在仓库中，因此本地测试期评分需要团队自行同步 `evaluation/*.csv`。

## 后续纯强化学习

纯 PPO 的状态、动作、官方同构奖励、16卡 rollout 规则和推荐超参数见
[RL/PPO 交接文档](docs/stockmixer_rl_ppo.md)及
[`configs/stockmixer_ppo_recommended.json`](configs/stockmixer_ppo_recommended.json)。它允许从当前 SFT
checkpoint 初始化，但 PPO 阶段所有监督辅助权重均为0。

## 第三方代码与许可

本项目原创代码使用 MIT License。`third_party/` 中的项目保留各自的上游版权和许可；引用信息、
固定 commit 与本地 patch 见 [third_party/README.md](third_party/README.md)。比赛数据和官方题目文件
不随本仓库再分发。
