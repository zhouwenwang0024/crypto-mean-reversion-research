# 两年新样本与滚动模型复核报告

本报告是交接包审计报告的补充，主结果使用原策略的 1 分钟时钟。交接包中的原始回放、止损边界和缓存身份问题见 [`REPORT_ROLLING_RIDGE_AUDIT_ZH.md`](REPORT_ROLLING_RIDGE_AUDIT_ZH.md)；本报告不把旧结果和新样本结果混在一起。

## 数据与时序

我复用了原来的 20 个 USDT 永续合约，不重复下载 2026-03 之后已有的 500 个分区。新增归档为 2024-01 至 2026-02，其中 2024-01 和 2024-02 只作预热，正式评估区间为 2024-03-01 至 2026-03-01，正好两年，与旧数据区间不重叠。

数据质量文件为 [`results/extended_data_quality.json`](results/extended_data_quality.json)。其中 520 个 1 分钟分区共 22,752,000 行，每币 1,137,600 行；时间网格、价格、成交量和 OHLC 关系均通过检查。1 分钟零报价量有 2,876 个，聚合后的零报价量 5 分钟柱有 420 个；它们保留在特征中，但入场要求前 5 分钟每个币有正报价量。新增的 20 个 2024-01 分区逐条完成官方 ZIP checksum 校验，后续运行显示 520/520 为 `existing`、0 次重新下载。数据内容摘要为 `fbb24c49342aad0928626200067cf8e28742360138cdf65b6de3928486024789`。

资金费单独来自 Binance `fundingRate` 接口，20 币各 2,277 条，共 45,540 条，覆盖 2024-02-01 至 2026-02-28，标记价均为有限正数，时间网格最大误差 16 ms；缺失不会补零。资金费在每个时间点先于同一时刻成交结算。

## 正式分钟验证

每个日内拟合只使用当日以前的 672 个完整小时收益。Ridge 使用标准化协方差和归一化惩罚 0.1；PCA 是逐目标留一币、只在过去样本标准化和拟合的 PCA2/PCA3/PCA5；`ridge_recent_hourly` 使用 168 小时半衰期的指数权重；`ridge_huber` 对每个目标做五轮残差降权。所有模型、币池、阈值和成本口径在最后六个月之前冻结。

参考价是过去 180 个 1 分钟 log-close 的因果中枢，尺度是当前权重投影后的前 7 个完整日偏离。每 5 分钟观察一次，信号使用完成的上一分钟，下一分钟开盘成交。入场为 `|z| >= 2` 且实际偏离至少 1.5%；动态退出带为约 0.25%，止损为含浮点容差的 3%，持仓期限在引擎中按成交到成交的 240 分钟边界处理；最多三仓、单仓 30% 当前权益、新仓总预算 90%。费用按同一时刻全部币种的有符号净订单额合并计费。

正式结果在 [`results/extended_minute_model_results.csv`](results/extended_minute_model_results.csv)，模型权重稳定性在 [`results/extended_minute_model_stability.csv`](results/extended_minute_model_stability.csv)，配置和数据摘要在 [`results/extended_minute_validation_manifest.json`](results/extended_minute_validation_manifest.json)。下表是最后六个月、单边 5 bp、无资金费的完整账户结果；开发段和验证段也在结果表中保留。

| 模型 | 收益 | 最大回撤 | 交易数 |
|---|---:|---:|---:|
| Ridge SMA3 | -7.09% | -38.44% | 950 |
| PCA2 SMA3 | -10.06% | -40.35% | 1,007 |
| PCA3 SMA3 | -23.23% | -50.21% | 998 |
| PCA5 SMA3 | -11.79% | -41.21% | 973 |
| 指数加权 Ridge | -17.93% | -42.21% | 949 |
| Huber Ridge | -2.05% | -37.74% | 946 |

加入资金费后同一段、同一成本的结果为：Ridge -7.02%（资金现金流 +0.077%）、PCA2 -10.08%、PCA3 -23.16%、PCA5 -11.63%、指数加权 Ridge -17.89%、Huber Ridge -1.89%（资金现金流 +0.195%）。Huber 是这组模型中相对最好者，但仍没有成本后正收益证据；10 bp 时 Huber 为 -26.26%。0/2/5/10 bp 全部实际运行，资金费结果另列 `funding=true`，没有把资金费当作零。

模型稳定性不能代替样本外收益。日权重 L1 变化均值分别为 Ridge 0.00726、PCA2 0.00242、PCA3 0.00358、PCA5 0.00526、指数加权 0.01217、Huber 0.00560；PCA 权重较平滑，但 PCA2/PCA3/PCA5 在成本后仍亏损。

## 机制与风险对照

冻结同一 Ridge 和同一 5 bp 口径的对照在 [`results/extended_minute_mechanism.csv`](results/extended_minute_mechanism.csv)。目标单腿在开发/验证/留出分别为 -47.89%/-50.15%/-7.09%；半对冲为 -65.22%/-36.08%/+8.07%；全对冲为 -65.90%/-32.84%/-5.08%。半对冲的正留出结果是在看到留出数据后进行的机制探索，不能称为新的样本外确认；它在前两段明显亏损，应冻结后再用完全未使用的新区间检验。留出额外成交延迟 2/3 分钟分别为 -11.95%/-25.32%，改为 15 分钟观察为 -37.96%。对冲降低了最大净方向敞口，但会增加交易腿和费用，不能用额外杠杆美化收益。

Ridge 留出 5 bp 的 950 笔订单证据在 [`results/extended_minute_ridge_holdout_5bp_orders.csv`](results/extended_minute_ridge_holdout_5bp_orders.csv)。按目标币和入场月汇总分别见 [`results/extended_minute_target_summary.csv`](results/extended_minute_target_summary.csv) 和 [`results/extended_minute_month_summary.csv`](results/extended_minute_month_summary.csv)。净收益 -7.09% 高度集中：SUI 单币贡献 +57.28%，LTC -15.94%、XRP -11.25%；2025-10 单月贡献 +38.06%，2026-02 为 -20.60%。这说明总体数字不能当作分散、稳定的跨币收益。

## 独立审计

[`results/extended_minute_causal_checks.json`](results/extended_minute_causal_checks.json) 是实际分钟引擎的独立核对：逐目标增广最小二乘 Ridge 的 `W` 最大误差 `6.8e-15`，中枢误差 `1.7e-13`，尺度误差 `1.5e-16`；独立数量和现金库存账本的权益误差 `8.9e-15`、数量误差为 0。未来价格后缀扰动时，所有前缀特征、权重、账户权益和持仓均不变，后缀权益变化为 1.7560。实际调用方向反转和漏扣费用两个负例，均被账本检查拒绝，`all_pass=true`。

这次迁移也定位了低频实验引擎的几个问题：旧 5 分钟脚本的小时采样早了 55 分钟，3 小时中枢没有使用全部分钟价格，旧对冲数量把 Ridge 系数当作价格加权数量，期末行政平仓没有写回权益序列，资金费事件数按持仓重复计数。旧的 `extended_model_results.csv` 仅保留作低频诊断，不作为正式两年结论；正式结论只引用 native-minute 文件和上面的独立审计。

限制仍然存在：成交模型是下一分钟开盘的研究账本，没有撮合队列、滑点、清算或保证 maker 成交；零成交量价格仍会参与历史特征；2024-03 至 2026-03 是本次真正冻结规则后的新样本，旧 2026-03 之后六个月仍属于既有研究区间。当前证据支持“模型和执行细节可复核、成本后优势不稳健”，不支持宣称稳定盈利。
