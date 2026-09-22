# FinAxial C0 checkpoints

This directory contains the three frozen FinAxial C0 checkpoints and compact evaluation records.

| Seed | Selected epoch/update | SHA-256 |
|---:|---:|---|
| 2026 | 30 / 180 | `3b186d87c1226ab793c755205f698a0974353441320f00d3e9c4cd812989b9c1` |
| 2027 | 30 / 180 | `5084c4ec4bdbd09a2d0e29f7d9c68ffc8da3ddf4234faa4ae2320b1973219e92` |
| 2028 | 30 / 180 | `88b634b73fa102ebca85a1918c5d2212d0bf233f8daffcabe02da2dd2262e6b2` |

Each seed directory includes the raw model weights, metadata, training summary, and validation
history. `evaluation/evaluation_results.json` contains the full 343-date test replay. Large prediction
arrays are intentionally excluded from version control.
