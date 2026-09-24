# 20 个币种均值回归与统计套利研究

正式样本为 20 个 Binance USDT 永续标的，覆盖 2026 年 3—8 月的 1 分钟数据。研究优先看 [REPORT_V3.md](REPORT_V3.md)；它在已审计的 v2 账本上比较了多种合理价、周期和开平仓规则，并报告了完整留出期结果。

v3 的主要代码是 `src/research_v3.py`，会读取本地 Parquet，构造 B0、等权 PEER、留一中位数信号、Ridge、PCA1/3/5 和 AR1 控制。每个配置只使用当前完成柱之前的数据，信号柱结束后延迟两分钟，用真实 1 分钟开盘成交，计入资金费和每腿 gross turnover 成本。

```powershell
python src/research_v3.py
python src/summarize_v3.py
python -m py_compile src/research_v2.py src/research_v3.py src/summarize_v3.py tests/test_research_v2.py tests/test_research_v3.py
python -m pytest -p no:pytest_anchorpy -p no:anyio -p no:requests_mock -p no:pytest_ethereum --assert=plain -q tests
```

v3 预先声明并去重后运行 67 个配置，主网格使用 2 bp gross 成本；5 个固定模型控制和 5 个验证期代表配置另做 0、1、2、3 bp 敏感性。最终结果没有显示稳定的成本后多标的优势，不能直接用于实盘。

主要结果文件：

- `REPORT_V3.md`：最新研究报告、公式、开平逻辑、审计与限制。
- `results/v3_grid_2bp.csv`：67 个配置的完整验证和留出结果。
- `results/v3_selected_costs.csv`：成本敏感性结果。
- `results/v3_model_summary_2bp.csv`、`results/v3_logic_summary_2bp.csv`：分模型和分规则汇总。
- `results/v3_bootstrap_ci.csv`：4 日区块 bootstrap 描述性区间。
- `results/v3_selected_*`：代表配置的交易、权益和账本。

旧的 v1 结果已在 `AUDIT.md` 中标记为无效；v2 报告保留用于审计历史，不应与 v3 的 residual-scale 和 gross-fee 口径直接比较。
