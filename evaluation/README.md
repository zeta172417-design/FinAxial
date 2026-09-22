# 本地测试期测评

本目录可单独同步给组员，用于对统一测试期上的预测文件评分。

## 数据定义

训练集验证表明，标签严格满足：

```text
y_ret_1d(t) = close(t+1) / close(t) - 1
```

因此 `build_test_labels.py` 使用同一股票下一交易日的 `close` 重构测试期标签。原测试集
最后一个交易日 `20260608` 没有后续价格，X 和 Y 都将该日整体剔除。最终本地测评数据为：

```text
2025-01-02 至 2026-06-05
343 个交易日 × 4,650 只股票 = 1,594,950 行
```

如果当前行或下一行的 `close` 缺失，对应 `y_ret_1d` 保持 NaN，参考脚本会在 Rank IC
和收益计算中自动过滤。

## 环境

推荐使用项目固定解释器：

```bash
cd /path/to/FinAxial
PYTHON=${FINAXIAL_PYTHON:-python}
```

仅同步 `evaluation/` 到没有该环境的机器时，需要 Python 3，以及 `numpy`、`pandas`、
`scipy`；`loguru` 是可选依赖。

## 直接运行测评

提交文件必须恰好包含以下三列：

```text
ts_code,trade_date,pred
```

并且必须覆盖本目录 Y 文件中的全部 1,594,950 个唯一键。运行：

```bash
$PYTHON -m evaluation.run_evaluate /path/to/submission.csv
```

如果当前目录只有同步得到的 `evaluation/` 文件夹，可在它的上一级执行同一命令；或者：

```bash
PYTHONPATH=/path/to/folder python /path/to/folder/evaluation/run_evaluate.py \
  /path/to/submission.csv --data-dir /path/to/folder/evaluation
```

入口会先检查列名、行数、重复键、缺失/无限预测以及完整键集合，再调用参考
`evaluate.py`，输出 Rank IC、ICIR、Top 10% 年化超额、换手率和综合分。

## 重新生成标签

项目完整目录中可执行：

```bash
$PYTHON evaluation/build_test_labels.py \
  --source data/raw/测试集_X.csv \
  --train-source data/raw/训练集.csv \
  --output-dir evaluation
```

`label_manifest.json` 记录源文件、公式、日期、行数、缺失标签数和 SHA-256。通常组员只需
使用已经生成好的 X/Y，无需重复生成。

## 重要口径说明

这种标签使用了测试日期之后的价格，是事后重构标签。它非常适合组内统一复核和分析，
但在线预测要求每个交易日只能使用当时可见数据，直接在这些标签上反复调参会产生未来
信息泄漏，得到的分数不能代表严格样本外能力。报告中应明确标注为“测试期事后重构评测”。
