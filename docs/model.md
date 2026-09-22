# FinAxial C0 architecture

## Input and causality

FinAxial C0 consumes the complete stock universe as one cross-sectional sample. Each input has
96 date tokens: 32 burn-in dates and 64 supervised dates. Every OHLCVA token is normalized with
statistics from its own trailing 64-date history only. Missing pre-listing history remains invalid;
other missing OHLC values use the previous close while volume and amount are zero-filled.

The model never treats the 64 outputs as unknown future dates. During training they are 64 known,
consecutive historical dates processed in parallel, and the causal temporal mask prevents an earlier
output from reading a later token. Live inference uses the last output available at the current date.

## Network

- Feature projection: 6 channels to `d_model=128`.
- Stock identity: 32-dimensional learned embedding projected to 128 dimensions and injected through
  a feature-dependent sigmoid gate. Ten percent of identities are replaced by an UNK embedding in
  training.
- Temporal attention: two sliding causal self-attention blocks, four heads, RoPE, window 64.
- Stock attention: two shared full-cross-section blocks without stock positional encodings.
- Axial order: Temporal 1, Stock 1, Temporal 2, then Stock 2 on the 64 supervised dates.
- FFN width: 512. Dropout: 0.1. Attention dropout: 0.
- Head: LayerNorm and a bias-free scalar projection followed by eligible-stock centering.
- Parameters: 964,064.

The absence of stock positional encodings makes stock attention permutation equivariant. Company
identity is carried by the explicit vocabulary embedding, whose ordered vocabulary SHA-256 is stored
with every checkpoint.

## Objective and optimization

The differentiable objective combines daily soft Rank IC, bounded soft Top-10% annualized excess,
and adjacent-date soft portfolio stability with weights 0.4, 0.3, and 0.3. Rank temperature anneals
from 0.20 to 0.05 and portfolio temperature from 0.04 to 0.01 during the first 20 epochs.

Training uses AdamW, learning rate `3e-4`, weight decay `0.05`, gradient clipping at 1.0, five linear
warmup epochs, and cosine decay to 1% of the initial rate. Checkpoints are selected by the trailing
three-evaluation mean of the exact raw composite score on the development period.

## Saved model family

The repository stores seeds 2026, 2027, and 2028. The recommended robust inference path converts
each model output to a same-date percentile rank and averages the three ranks. A fixed causal signal
EWMA with alpha 0.25 is reported separately and is not part of the checkpoint weights.
