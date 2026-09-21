# Stock-Time Transformer B0

## 固定架构

- 输入：4,650只股票，96个日期，6个OHLCVA通道。
- 因果归一化：每个token独立使用截至自身日期的trailing-64统计量。
- Burn-in：前32个日期不直接计算loss。
- 并行监督：后64个日期输出`[64, 4650]`收益排序信号。
- Temporal Encoder：2层、`d_model=96`、4 heads、FFN 384、RoPE、64-token sliding causal mask。
- Stock Encoder：每日共享2层全截面self-attention，无股票位置编码。
- 公司身份：32维可学习stock embedding，训练时10%替换为UNK。
- 输出：LayerNorm和无bias线性头；每日在eligible股票上做截面中心化。

Stock Attention参数由所有日期共享。模型不会一次预测未知的未来64天；训练时64个输出对应64个
已有行情日期，并且每个输出只能访问自身及更早token。真实推理只使用截至当前日的最后一个输出。

## 数据与mask

- OHLC停牌缺失使用此前最近收盘价，vol/amount置零；上市前保持无效。
- 标签为`close(t+1) / close(t) - 1`。
- Rank IC使用全部有标签股票。
- Top 10%收益和换手率排除涨停股票，与官方评分实现一致。
- 每个block包含63个相邻日期soft Jaccard，不跨股票或日期拼接序列。

## 训练设置

- AdamW，LR `3e-4`，weight decay `0.05`。
- 5 epochs线性warmup，其后cosine decay至基础LR的1%。
- 梯度裁剪1.0，30 epochs，8卡DDP。
- 训练stride 32，共51个block；drop-last后每卡每轮6次更新。
- 参数EMA关闭；raw Final连续3轮均值选择checkpoint。
- SwanLab项目：`financial-modeling-stock-time-transformer`。

选定checkpoint是epoch 23（optimizer update 138），峰值PPU显存约20.71 GiB。

## 为什么保留B0

完整343日评测中，B0 Final为0.328791。增加后置Temporal Decoder的B1只有0.320384；它虽可提高
长历史位置的Rank IC，却导致换手恶化。严格同日期对照进一步显示B0在约64日历史处饱和，因此正式
模型保留B0且不继续扩大时间窗口。
