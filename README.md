# FinAxial

FinAxial 是一个两阶段的金融截面建模框架：双头轴向 Transformer 预测器学习股票排序与次日收益率，循环决策器通过 GRPO 学习收益差门槛和换股预算，在收益与持仓稳定性之间作出决策。

本公开仓库仅维护 FinAxial 模型及复现入口。已发布预测器、决策器、因子尺度校准参数和权重校验清单；不包含原始行情、外部标签、个人凭据、内部实验日志或实验分支。使用者需要自行合法获取数据。本项目不构成投资建议。

## 模型结构

```text
每日 OHLC、成交量、成交额 [N,T,6]
       + 因果衍生因子      [N,T,122]
       + 股票 ID          [N] → Embedding32
                         │
                  Linear128 → 128
                  + 门控股票身份向量
                         │ [N,T,128]
           ┌─────────────▼───────────────────────┐
           │ 交错轴向 Transformer × 2            │
           │ 同股时间 attention：RoPE、causal    │
           │ 同日股票 attention：无位置编码       │
           └─────────────┬───────────────────────┘
                         │ hidden [T,N,128]
             ┌───────────┴───────────┐
             ▼                       ▼
      LN → Linear(128,1)      LN → Linear(128,1) × 0.02
      排序分数 [T,N]           次日收益预测 [T,N]
             └───────────┬───────────┘
                         │ 冻结预测器
           ┌─────────────▼───────────────────────┐
           │ 市场 hidden 均值 → 16维市场摘要     │
           │ 持仓与高分候选 → 133维逐股token     │
           │ → 64维编码 → 4 query注意力 → 16维   │
           │ 市场＋组合摘要 → 拼接6维统计量      │
           │ → GRUCell(22,32) → 高斯动作均值[2]  │
           └─────────────┬───────────────────────┘
                         ▼
                收益差 margin／换股 budget
                         ▼
          昨日持仓 → 因果换股 → Top 10%与完整排序
```

`N` 为固定股票词表大小，发布权重对应4,650只股票；`T=256`，hidden为128维，4个注意力头，FFN为512维，股票身份向量为32维。股票槽位顺序通过SHA256校验，不允许任意重排。

原始6维使用最近64个交易日的逐股因果z-score。122维附加因子包含价格形态、动量、波动、量额、截面排名和市场摘要，部分历史周期达到252日。连续因子采用训练期拟合的稳健尺度参数，截面排名保留其原尺度；推理不重新拟合校准参数。完整定义见 `finmodel/factors.py`。

预测器的排序头优化可微 daily Rank IC，收益头优化固定尺度 Huber，两种梯度都进入共享主干。训练256日窗口、步幅32、20 epoch；AdamW、LR `3e-4`、5 epoch warmup后cosine。原始行情归一化的64日窗口不等于Transformer的256日感受野。

决策器直接读取冻结预测器的hidden与两个输出，加上昨日持仓、持仓年龄等状态。margin比较的是预测收益率差，不是排序分数差；budget控制主动替换数量，被动替换按可交易性处理。没有alpha或EWMA。训练采样高斯动作；推理使用均值，并连续传递GRU、持仓与年龄。

换入候选按原排序从高到低、原持仓按原排序从低到高配对，在budget允许的配对中逐一检查预测收益差是否超过margin。目标持仓确定后，仅将需要跨越Top10%边界的股票分数抬升／压低到边界附近，保留两组内部的原相对顺序及不必调整的分数。该局部投影尽量减少排序扰动，但不保证数学意义上最小的Rank IC损失。

GRPO每日奖励为：

```text
0.4 × Rank IC + 0.3 × 252 × 当日Top10%超额收益
              + 0.3 × 持仓Jaccard稳定性
```

每天在64条轨迹内做优势标准化，不使用critic或参考KL惩罚。每次采集64条完整轨迹，分4批更新，每批16条、每条使用一次。训练预热32日＋计奖64日、步幅32；AdamW、LR `1e-4`、1 epoch warmup后cosine。报告复现的决策器训练8 epoch；完整训练集发布权重保留原10 epoch。两套预测器与决策器均使用seed2026，报告复现另固定动作采样噪声seed2026。初始margin为0.002、budget为100、动作标准差为0.4。验证／推理预热64日，预热不计分。所有训练统一保留最后完成epoch，不用验证集挑checkpoint或早停。

## 安装与权重

Python ≥3.10。先在自己的隔离环境安装适合硬件的PyTorch，再安装本项目：

```bash
python -m pip install -e .
python scripts/verify_weights.py
python -m unittest discover -s tests
python scripts/smoke_weights.py --device cuda:0
```

PPU用户应使用厂商适配Torch，并在启动前执行 `source /usr/local/PPU_SDK/envsetup.sh`。不要覆盖共享环境中的torch、torchvision、torchaudio。本仓库不提供厂商驱动；不同硬件、依赖版本与非逐位确定的算子可能造成数值差异。

权重随仓库直接提供，不需要额外下载大数据集：

```text
weights/
├── predictor/model.pt、metadata.json
├── decision/policy.pt、metadata.json
├── factor_calibration.json
├── score_calibration.json
└── manifest.json
```

`weights/` 是训练全历史数据的发布模型；内部验证的模型需要按下一节在历史子集上重新训练，不能用这些全历史权重报告内部验证成绩。

## 复现内部时间留出验证

输入训练CSV包含 `ts_code, trade_date, open, high, low, close, vol, amount, flag_limit_up, flag_limit_down, y_ret_1d`。

默认复现协议：2018–2023作为训练历史，2024作为242日内部验证。2023-12-29样本从训练排除，避免其次日标签跨入验证期；因子校准只使用训练区间。原始CSV不修改，运行目录拒绝覆盖未完成结果。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/reproduce.sh data/train.csv
```

脚本顺序执行：面板 → 训练期因子校准 → 20 epoch预测器 → 冻结逐日缓存 → 8 epoch GRPO → 最后checkpoint验证。训练配置位于 `configs/predictor.json`、`configs/decision_grpo.json`；默认不启用外部日志服务。

该配置的本地参考结果如下。2024年用于内部配置评估，不是独立盲测，也不是未知标签测试成绩：

| 方法 | 总分 | Rank IC | 年化超额收益 | 换手率 |
| --- | ---: | ---: | ---: | ---: |
| 预测器直接排序 | 0.2607 | 0.1115 | 50.84% | 78.80% |
| FinAxial预测器＋决策器 | 0.4523 | 0.1073 | 42.08% | 5.62% |

无需获得测试集标签即可完成上述复现。由于股票身份词表、数据、依赖和硬件会影响结果，更换数据后不能预期复现相同数值。

## 完整测试集推理：无需标签

特征CSV只需要主键、原始6维行情与涨跌停标志。推理会使用此前训练历史进行归一化、长期因子计算及决策预热；不会读取测试标签，也不会剔除最后一个特征日期。

```bash
python scripts/preprocess.py --source data/train.csv --output artifacts/history
python scripts/predict_final.py \
  --history-panel artifacts/history --features data/test_features.csv \
  --expected-days 344 --device cuda:0 \
  --score-calibration weights/score_calibration.json \
  --output artifacts/submission.csv
```

使用其他长度的数据时删除或修改 `--expected-days`。输出严格为 `ts_code,trade_date,pred`，覆盖全部输入日期／股票键，包括最后一天。344日×4,650只股票应为1,599,600行；伴随manifest记录模型哈希、覆盖率检查与无标签推理声明。上述命令导出的 `pred` 已按训练标签校准为小数收益率尺度，不是Rank IC指标，也不是独立收益头的直接输出。

也可以先导出原始排序信号，再在CPU上执行校准与完整格式核查：

```bash
python scripts/predict_final.py \
  --history-panel artifacts/history --features data/test_features.csv \
  --expected-days 344 --device cuda:0 \
  --output artifacts/raw_predictions.csv
python scripts/calibrate_submission.py \
  --raw artifacts/raw_predictions.csv --features data/test_features.csv \
  --calibration weights/score_calibration.json \
  --expected-days 344 --output artifacts/submission.csv
```

导出端执行全局正斜率仿射变换 `pred = a × decision_score + b`，发布参数为 `a=0.008105780947825202`、`b=-0.0038215032924382424`。参数仅用2018–2024训练期已知标签拟合，与发布权重哈希绑定；不使用推理期标签、不裁剪、不改收益头、margin或持仓。正斜率变换保持排名、Top10%及排名型评分，CSV采用17位有效数字以减少精度损失。它只是把组合排序信号校准到小数收益率尺度，并不是独立收益头的预测，也不保证未来收益幅度准确。内部留出验证若需校准，应另用该次训练子集拟合，不能直接使用这份全历史参数。

冻结预测器以重叠窗口生成因果输出，决策器在整个推理期连续自回归运行，不按训练窗口重置。窗口对齐、状态预热与可评分日期均需固定；扩大日期范围时重新切窗口可能改变原有日期的模型信号，不能把这种差异归因于输出校准。已发布的逐日信号应保持固定，新增日期延续决策状态；对同一份原始信号施加正斜率校准时，排序型指标保持不变。需要较多显存的预测器推荐在经过验证的PPU环境或具备足够显存的CUDA设备运行；`--device cpu`可用于小规模功能测试，无静默CPU回退。

若自行持有合法标签，可以独立评分：

```bash
python scripts/evaluate_predictions.py \
  --predictions artifacts/submission.csv \
  --features data/test_features.csv --labels data/evaluation_labels.csv
```

评分复刻Rank IC、Top10%年化超额收益与Jaccard换手率的加权公式：IC使用有标签股票；收益组排除涨停与缺失标签；换手组排除涨停；不额外过滤跌停，不对预测值进行涨跌停裁剪。没有标签的最后日期可以保留在预测文件中，但不参与有标签评测。不提供从测试价格反推标签的脚本。

## License

原创代码采用MIT许可，见 `LICENSE`。外部数据的使用与再分发应遵循数据提供方的授权。
