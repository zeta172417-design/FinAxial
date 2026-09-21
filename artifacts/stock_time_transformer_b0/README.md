# Stock-Time Transformer B0 checkpoint

This directory contains the selected reproducible checkpoint and compact evaluation records.

- Architecture: Temporal Encoder -> Stock Attention -> prediction head
- Context: 32 burn-in days + 64 supervised days
- Selection: highest three-epoch trailing mean of raw validation final score
- Selected epoch/update: 23 / 138
- Parameters: 609,728
- `best/model.pt` SHA256: `5076c4560ccc0b2f7d4238515423eaa145ef2e335e58a24c06553f5d0980c891`

Raw validation metrics: Final `0.328791`, Rank IC `0.054986`, annual excess
`0.158661`, stability `0.863996`. The optional fixed EWMA alpha 0.25 reaches
Final `0.336329`; the distributed checkpoint itself contains raw model weights.
