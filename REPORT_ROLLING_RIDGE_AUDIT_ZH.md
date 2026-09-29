# Rolling Ridge 候选复现、审计与小范围改进

## 范围与数据

本报告替代仓库中早期“压缩包缺失”的 Ridge 报告。交接包 `rolling_ridge_calculation_and_audit.zip` 已找到并读取 `STRATEGY_SPEC.json`、`README.md`、`REPORT_ZH.md`、`work/src` 和结果文件。仓库本地 120 个分钟 Parquet 的数据包 SHA-256 为 `552934b51b0c0625119dc40bfa89893c174aae30a1e7b8ea0d397bb981f7b870`，与交接报告一致；没有重新下载行情。

正式候选是 `ridge_sma3/live_gap25`：过去 672 个完整小时收益逐日拟合标准化 Ridge，协方差惩罚为 0.1；3 小时分钟 log-close 中枢排除当前观测；sigma 是当前权重投影后过去 7 个完整日的分钟偏离标准差；每天重投影历史特征。每 5 分钟用 `close[t-1]` 做信号和数量，`open[t+1]` 成交；正残差做空、负残差做多；只持有目标币，最多三仓、单仓当前权益 30%、新增预算 90%。每分钟动态 gap 退出，最长 240 分钟，实际价格亏损达到 3% 触发延迟退出，期末按最后分钟收盘行政结算。主表不计资金费，费用按真实净订单名义额收取。

可运行入口在 [`src/rolling_ridge_audit/replay.py`](src/rolling_ridge_audit/replay.py)，原始和止损容差修正版引擎分别保存在 `engine_original.py` 与 `engine.py`。特征缓存现在写入被忽略的 `tmp/rolling_ridge_cache`，并把完整模型规格写入缓存；规格不一致时自动重建，避免同名旧缓存污染。

## 复现结果

结果由 [`results/rolling_ridge_corrected.csv`](results/rolling_ridge_corrected.csv) 和 [`results/rolling_ridge_raw.csv`](results/rolling_ridge_raw.csv) 生成。两段均空仓启动：

| 区间 | 单边费用 | 原始止损边界 | 修正版收益 | 交易数 | 修正版最大回撤 |
|---|---:|---:|---:|---:|---:|
| 2026-05-01/2026-07-01 | 0 bp | +37.592992% | +37.592992% | 475 | −6.9162% |
| 2026-05-01/2026-07-01 | 2 bp | +29.984880% | +29.984880% | 475 | −7.4637% |
| 2026-05-01/2026-07-01 | 5 bp | +19.352006% | +19.352006% | 475 | −8.2857% |
| 2026-05-01/2026-07-01 | 10 bp | +3.522891% | +3.522891% | 475 | −10.4433% |
| 2026-07-01/2026-09-01 | 0 bp | +24.957496% | **+24.855839%** | 402 | −8.6264% |
| 2026-07-01/2026-09-01 | 2 bp | +19.078753% | **+18.981860%** | 402 | −9.8000% |
| 2026-07-01/2026-09-01 | 5 bp | +10.773601% | **+10.683426%** | 402 | −11.5390% |
| 2026-07-01/2026-09-01 | 10 bp | −1.803459% | **−1.883469%** | 402 | −16.1007% |

这复现了交接包的 0/5/10 bp 核对值。独立 beta/参考价计算采用每个目标单独的最小二乘增广系统，154 个日快照、3080 个目标拟合与主特征的最大误差为：`dev 5.62e-11`、`center 5.62e-11`、`scale 2.05e-12`、`W 4.61e-15`。独立现金账本没有调用引擎的 PnL 汇总，5–6 月 NAV 最大误差 `1.44e-14`、数量误差 `4.17e-14`；7–8 月分别为 `1.20e-14`、`4.62e-14`。明细在 [`results/feature_crosscheck_local.json`](results/feature_crosscheck_local.json) 和 [`results/independent_accounts_local.json`](results/independent_accounts_local.json)。

## 发现的问题

第一处实际经济结果差异是 3% 止损的浮点等号边界。AAVE 在 2026-08-21 23:54 的数学亏损为 3%，浮点表示是 `-0.029999999999999943`，原始 `<= -0.03` 没有触发，下一分钟才退出；容差 `1e-12` 使它提前一分钟退出。交易数仍为 402，但 7–8 月 5 bp 收益下降 0.090175 个百分点。证据在 [`results/stop_boundary_local.json`](results/stop_boundary_local.json) 和 [`results/stop_replay_local.csv`](results/stop_replay_local.csv)。这是边界定义修正，不是收益优化，也不意味着 3% 是最大损失保证。

第二处是缓存身份缺失。交接包的反例显示同一个模型名在请求 5 小时特征时仍返回旧的 3 小时缓存。本次正式结果均强制清洁重建；仓库实现把规格标签写入缓存并校验，参数不一致即重建。原包的 `cache_finding.json` 保留了该缺陷证据。

因果和合约测试实际调用引擎函数：未来后缀扰动在 2026-05-21 12:00、2026-07-15 00:00 之前的 dev、W、权益、数量误差均为 0，扰动后续权益分别改变 0.3421、0.5980；原始引擎的报价单位不变性最大权益误差为 `9.25e-4`，容差修正后为 `7.33e-15`。15 项核心测试全部通过，包含方向、动态退出、成交时序、240 分钟期限、费用、未来价格隔离和独立账本。故意反转方向、漏扣费用、使用未来价格定量三个反例均被捕获，汇总见 [`results/contract_checks_local.json`](results/contract_checks_local.json) 与 [`results/tests6_initial.json`](results/tests6_initial.json)。

默认 pytest 收集会触发环境中旧版 `anchorpy` assertion hook 的兼容性错误；按仓库命令禁用这些插件并使用 `--assert=plain` 后，`tests/test_ridge_candidate_audit.py` 的 5 项和 `tests/test_research_v3.py` 的 7 项均通过。

## 资金费与成本承受力

`results/funding_api` 有 20 币、11040 条结算记录，rate 和 mark 均为有限值；资金费没有默认为零。使用官方 mark 重放修正版 Ridge、5 bp：5–6 月收益 `19.403141%`，资金现金流 `+0.050837%`；7–8 月收益 `10.759438%`，资金现金流 `+0.068443%`。结果在 [`results/funding_replay.csv`](results/funding_replay.csv)。这仍是分钟级成交代理，未模拟真实队列、深度、滑点或清算。

## PCA 与 Ridge 对照

[`results/pca_ridge_package_compare.csv`](results/pca_ridge_package_compare.csv) 使用同一交接引擎、同一门槛和同一费用，PCA 只用过去 28 天其他 19 币的标准化小时收益，保留 2、3、5 个因子，并使用相同 0.1 归一化惩罚。所有模型没有资金费：

| 模型 | 5–6 月 0 bp | 5–6 月 5 bp | 7–8 月 0 bp | 7–8 月 5 bp | 7–8 月最大回撤(5 bp) |
|---|---:|---:|---:|---:|---:|
| Ridge | +37.5930% | +19.3520% | +24.8558% | +10.6834% | −11.5390% |
| PCA2 | +28.7291% | +11.5226% | +21.3472% | +7.5146% | −12.6358% |
| PCA3 | +28.6079% | +11.3801% | +13.1021% | +0.2635% | −15.6211% |
| PCA5 | +31.2513% | +13.7735% | +31.3080% | +16.2049% | −9.4753% |

PCA5 的日间权重漂移均值为 0.0405、最大单一 peer 系数中位数 0.1184，Ridge 为 0.0665、0.2097；PCA2/PCA3 更稳定，但成本后收益更弱。PCA5 在已反复研究过的 7–8 月历史区间表现较好，不能称为新的样本外验证；收益集中在少数币，5 bp 净收益贡献最大的包括 NEAR、AAVE、HBAR、SUI，明细见 [`results/model_target_5bp_summary.csv`](results/model_target_5bp_summary.csv)。因此保留 PCA5 作为历史研究候选和稳定性对照，不替换已审计 Ridge 基准，也不宣称盈利已被证明。

Ridge 惩罚邻近值 0.05、0.1、0.2 的严格引擎比较在 [`results/ridge_penalty_package_compare.csv`](results/ridge_penalty_package_compare.csv)。0.2 在验证段较高，但 5 bp 留出为 10.5744%，略低于基准 10.6834%；0.05 留出为 10.3209%。没有足够证据改变 0.1 基准。仓库已有的 `ridge_lambda_scan.csv` 属于另一套 v2 时钟，仅作补充，未被拿来宣称候选复现。

## 结论与限制

原候选可以在本地数据上复现，且独立 beta、参考价和现金数量账本一致。需要修正的是真实数值边界和缓存身份，而非方向、跨日坐标、未来信息、订单数量或动态退出主逻辑。修正后策略在 5 bp 成本下两段历史仍为正，但 10 bp 的 7–8 月为负，回撤约 8.6%–16.1%，收益和风险集中于少数币。PCA5 提供了较稳定的历史对照，尚不足以证明更强的未见样本外表现。下一次真正的新区间评估前应冻结模型、币池、门槛、费用和成交延迟；本报告没有把已经反复研究的 7–8 月称为新样本外。


---

Two-year native-minute validation, six-model comparison, and independent ledger audit: [`REPORT_EXTENDED_VALIDATION_ZH.md`](REPORT_EXTENDED_VALIDATION_ZH.md).
