# Local market data

This directory contains local inputs and is excluded from version control.

| File | Bytes | Date coverage | Purpose |
|---|---:|---|---|
| `训练集.csv` | 946,832,276 | 2018-01-02 to 2024-12-31 | Labelled training history |
| `测试集_X.csv` | 168,825,396 | 2025-01-02 to 2026-06-08 | Unlabelled feature history |

The preprocessing pipeline reads these files without modifying them. The feature file contains
1,599,600 rows, 4,650 stocks, and 344 dates. Generated labels, panels, and predictions remain local
under `evaluation/` and `artifacts/`.
