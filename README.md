# 20 个币种均值回归研究

这是一个离线、可复核的 Binance USDT 线性永续均值回归研究项目。最终研究池为 20 个标的、2026-03 至 2026-08 的 1 分钟数据。TONUSDT 在样本后段停牌，按预先规定的覆盖和形成期成交额规则替换为 FILUSDT，证据见 `results/universe_replacement.json`。

最终修正版报告见 [`REPORT_V2.md`](REPORT_V2.md)。旧版回测曾有固定仓位、执行价前视和共享对冲腿成本问题；旧结果已在 [`AUDIT.md`](AUDIT.md) 中标记为无效，不应与 v2 结果混用。

## 运行

环境为 Python 3.10，依赖见 `requirements.txt`。本项目不需要 Rust/Cargo。

```powershell
python src/research_v2.py
python -m py_compile src/research_v2.py tests/test_research_v2.py
python -m pytest -p no:pytest_anchorpy -p no:anyio -p no:requests_mock -p no:pytest_ethereum --assert=plain -q tests
```

`research_v2.py` 会读取本地 Parquet，按 5、15、60 分钟构造完成柱，使用只依赖过去数据的 B0、等权 PEER（B1 类）和留一标的 PCA3（B2 类）。订单在信号柱结束并等待一个完整分钟后成交；现金账本在真实成交价、净订单成本和资金费结算后更新。主研究固定 29 个配置，先用 5—6 月验证，再单独报告 7—8 月留出期。

## 结果文件

- `REPORT_V2.md`：修正版中文报告与限制。
- `results/v2_grid_2bp.csv`：29 个预先固定配置的完整网格。
- `results/v2_selected_costs.csv`：一个验证期主配置与两个固定多标的对照，在 0—3 bp 成本下的验证/留出结果。
- `results/v2_monthly_equity.csv`、`results/v2_by_target_2bp.csv`：逐月和逐标的贡献。
- `results/v2_conditionals.csv`：按偏离幅度、方向和未来持有期的描述性条件收益。
- `results/v2_trades_rank*_2bp.csv`、`results/v2_ledger_rank*_2bp.json`：逐笔交易与执行账本。
- `results/funding_api/`：官方资金费率、结算时间和 markPrice；`results/funding_api_status.json` 保存抓取状态。

结论是验证期的正结果没有在留出期延续，多标的 PEER/PCA 对照没有显示稳定的成本后优势。因此这些结果适合继续延长样本和预注册规则，不适合直接进入实盘。
