# StockMixer 当前最优 SFT

唯一主版本为 `lookback=64`、20 epochs、16 卡 DDP 的 final-only SFT：

```text
soft_final = 0.4 * soft_rank_ic
           + 0.3 * bounded_soft_annual_excess
           + 0.3 * soft_top10_jaccard
loss = -soft_final
```

`best/model.pt` 是按 EMA 精确验证总分的5-epoch移动平均选出的 epoch 20 checkpoint。
验证 `final_score=0.3239581266`，文件 SHA256 为
`10692c8145083699675c7084379d322d569862ba26e0764ad0d2c8bb2c2d854c`。

`results/final_only_lb64_ep20/` 保存完整摘要、验证历史和训练日志。MSE/MAE 虽然仍被记录用于
诊断，但 `mse_weight=0`，不参与优化。
