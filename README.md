# 20 币种均值回归研究

本目录是独立的离线研究项目，研究币安 USDT 线性永续的 1 分钟成交 K 线。原始池为 20 个预先指定合约；检查发现 TONUSDT 在 2026-06-23 之后停牌，7、8 月为零成交且价格不变，因此按形成期成交额和完整覆盖选择 FILUSDT 替换，并在 `results/universe_replacement.json` 留痕。最终仍为 20 个币、6 个月、2026-03-01 00:00 至 2026-09-01 00:00 UTC。

回测净值部分目前处于审计状态：已发现固定仓位、B1 方向、延迟成交计账和共享净订单问题。请先阅读 [AUDIT.md](AUDIT.md)；现有收益图不能当作最终策略结论。

环境是 Python 3.10.9，依赖见 `requirements.txt`。本机没有 cargo，未做 Rust 编译。实际运行顺序：

```powershell
python src/research.py metadata       # 公开 exchangeInfo；失败会保留证据
python src/research.py download --sample --workers 3
python src/research.py download --workers 4
python src/research.py funding
python src/research.py quality
python src/research.py analyze
python src/research.py explore
python -m pytest -p no:pytest_anchorpy -p no:anyio -p no:requests_mock -p no:pytest_ethereum --assert=plain -q
```

本机直接 `python -m pytest -q` 会被全局 anchorpy 等插件与旧版 pytest 的 AST 重写兼容错误中断；上面的隔离插件命令实际通过 10 个测试。

行情月包先写 `.part`，校验官方 `.CHECKSUM` 后转成 ZSTD Parquet，并回读验证后删除临时 ZIP。`data/` 和 `results/` 是本地产物，不应提交大型原始文件。`results/source_manifest.jsonl` 保存 URL、HTTP 状态、下载字节数、SHA-256、行数、覆盖范围和异常。分析使用 5 分钟完整 K 线；信号在 5 分钟结束后等待 1 分钟，以随后 1 分钟开盘价执行。`explore` 运行预先声明的入场、持仓和 PCA 因子消融，并输出参数、条件收益和 PCA 诊断结果。

资金费使用官方 fundingRate 月度归档，实际结算费率和时间已保存；归档没有 markPrice，fapi 接口又在本机超时，所以交易表中的 funding 暂为未知项，净收益结论只能称为“未计 markPrice 资金费的毛/成本后结果”。
