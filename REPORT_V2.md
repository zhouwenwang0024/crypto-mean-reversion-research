# 修正版均值回归与统计套利研究报告

## 结论

本报告使用修正后的事件账本重新回测。最终 20 个 Binance USDT 线性永续、2026-03-01 至 2026-09-01 UTC 的 1 分钟数据完整覆盖 5,299,200 行。原始指定池的 TONUSDT 在 6 月下旬停牌，7、8 月无成交，因此按形成期候选池成交额和覆盖情况以 FILUSDT 替换；替换证据仍见 `results/universe_replacement.json`。

预先固定 29 个配置，用 3—4 月形成、5—6 月验证，最后只看一次 7—8 月留出期。验证期排名第一的是单标的 B0 基线：15 分钟信号、4 小时最大持仓、`|z|>=2` 入场、`|z|<=0.5` 平仓。它在 2 bp 单边成本下验证期上涨 2.08%，但留出期下跌 0.47%；0、1、2、3 bp 的留出收益分别为 −0.22%、−0.34%、−0.47%、−0.60%，日 Sharpe 分别为 −1.06、−1.26、−1.45、−1.65。固定的多标的 PEER（任务中的 B1 类方法）和 PCA3（任务中的 B2 类方法）对照在留出期更差。

因此，本样本没有支持可部署的稳定多标的均值回归优势。验证期的正结果没有延续到留出期，且结果对交易成本敏感；这也是本次研究的主要结论，而不是需要被隐藏的失败结果。

## 数据与时间切分

- 市场：Binance USDT 线性永续；UTC；不使用 2026 年 9 月数据。
- 最终 20 个标的：BTC、ETH、BNB、SOL、XRP、DOGE、ADA、TRX、LINK、SUI、AVAX、LTC、BCH、DOT、HBAR、XLM、FIL、UNI、NEAR、AAVE。
- 每个标的 264,960 个 1 分钟分区，合计 5,299,200 行；没有重复、乱序、分钟缺口、NaN 或坏 OHLC。零成交分钟不触发新信号，并且对应聚合柱被排除。
- 形成期：3—4 月；验证期：5—6 月；留出期：7—8 月。5 月 1 日和 7 月 1 日在账本中强制平仓，避免上一阶段持仓带入下一阶段。
- 真实资金费从 `/fapi/v1/fundingRate` 分页取得，使用实际 `fundingTime`、`fundingRate`、`markPrice`。正费率时多头现金流为 `-q * markPrice * rate`。资金费单独列示，并计入账户权益。

## 信号模型

所有中心和尺度都只使用当前信号柱之前的数据，滚动尺度为 7 天，中心为 4 小时 EWMA。

- **B0**：单标的自身对数价格相对因果 EWMA 中心的偏离；它是方向性基线，不是多标的统计套利。
- **PEER**（任务中的 B1 类方法）：目标标的对数价格减去其余 19 个标的等权对数价格。正偏离时做空目标、做多对冲篮子，组合按美元中性投影。
- **PCA3**（任务中的 B2 类方法）：每天 UTC 00:00 用此前 28 天的 5 分钟收益拟合 PCA；每个目标币用留一标的的其余 19 个币拟合因子，去除 3 个共同因子后累积残差。因子和对冲权重当天冻结，不看当天或未来收益。PCA1、PCA5 只作为预先固定的因子数对照。

主网格为 B0、PEER、PCA3 × 5、15、60 分钟 × 1、4、12 小时最大持仓，共 27 个配置；另固定 15 分钟/4 小时的 PCA1、PCA5 两个对照，总计 29 个。中心、尺度、入场阈值 2σ、退出阈值 0.5σ 和 3% 单笔组合止损在网格外冻结。

信号柱完整结束后再等待一个完整分钟：5 分钟信号在 `[t,t+5)` 柱结束后，于 `t+6` 的 1 分钟开盘成交；15、60 分钟按同一规则延迟。下单数量只用信号收盘价估算，实际 PnL 使用延迟后的真实开盘成交价。每个组合名义为当时权益的 10%，最多 3 个组合，单币净名义不超过权益的 20%。同一执行时刻先汇总每个币的净数量，再按净订单换手计成本，避免共享对冲腿重复收费。

## 回测结果

完整网格和固定对照的数字见 [`results/v2_grid_2bp.csv`](results/v2_grid_2bp.csv) 和 [`results/v2_selected_costs.csv`](results/v2_selected_costs.csv)。主配置与两个固定对照在 15 分钟/4 小时下的留出结果如下：

| 配置 | 0 bp | 1 bp | 2 bp | 3 bp |
|---|---:|---:|---:|---:|
| B0 主配置 | −0.22% | −0.34% | −0.47% | −0.60% |
| PEER 对照 | −0.93% | −1.06% | −1.19% | −1.32% |
| PCA3 对照 | −0.36% | −0.50% | −0.64% | −0.77% |

2 bp 下，主配置验证期为 +2.08%、67 笔交易，留出期为 −0.47%、63 笔交易、日 Sharpe −1.45、Profit Factor 约 0.87。主配置的月度权益变化为：3 月 −0.35%、4 月 +0.18%、5 月 +0.52%、6 月 +1.29%、7 月 +0.51%、8 月 −0.98%。按标的的盈亏集中且方向不稳定：AAVE、LINK、AVAX 分别约 +832、+776、+764 USDT；BCH、DOGE、NEAR 分别约 −864、−848、−679 USDT。逐月和逐标的明细见 [`results/v2_monthly_equity.csv`](results/v2_monthly_equity.csv) 与 [`results/v2_by_target_2bp.csv`](results/v2_by_target_2bp.csv)。

5 分钟收益的共同因子诊断显示，PC1 在形成、验证、留出期解释率为 70.7%、66.9%、60.4%，前三个因子为 78.8%、75.3%、68.3%。共同因子结构随时间下降，PCA 有诊断价值，但 PCA3 对照没有带来可交易增益；诊断数据见 [`results/v2_factor_diagnostics.csv`](results/v2_factor_diagnostics.csv)。条件未来收益按偏离方向、1.5—2、2—3、3—4、4σ以上和 15/60/240/1440 分钟分层，见 [`results/v2_conditionals.csv`](results/v2_conditionals.csv)。这些是重叠事件的描述性标签，不能当作独立交易收益或参数选择依据。

## 账本审查与测试

旧版回测曾固定仓位、错误使用未来成交价、反转 B1 方向，并重复计算共享对冲腿成本；旧结果已在 [`AUDIT.md`](AUDIT.md) 中标为无效，旧 `experiment_summary.csv` 等文件只作历史留痕。修正版使用显式持仓 lot、现金、实际成交价、净订单、资金费和边界平仓。主配置 2 bp 的逐笔 `net_pnl_including_funding` 合计与最终权益减初始权益的误差小于 `3e-11`。

测试命令：

```text
python -m py_compile src/research_v2.py tests/test_research_v2.py
python -m pytest -p no:pytest_anchorpy -p no:anyio -p no:requests_mock -p no:pytest_ethereum --assert=plain -q tests
```

结果为 **17 passed**。测试覆盖未来开盘价不应改变信号前的权益、延迟成交、正偏离做空目标、美元中性、聚合净换手、资金费结算先于同刻成交、空信号不交易、滚动中心的因果性和账本成本。

## 研究边界

样本只有六个月，留出期只有两个月；分钟事件高度重叠，不能把交易笔数当作独立样本。成本是每个净订单的全包单边 bps，没有单独模拟盘口冲击或真实 maker/taker 返佣。资金费 markPrice 来自公开接口，仍可能与具体账户实际成交存在差异。结果支持继续收集更长样本、预先注册规则并做滚动留出验证，不支持当前结果直接进入实盘。

方法背景可参考 [Gatev、Goetzmann、Rouwenhorst 的 pairs trading 框架](https://www.nber.org/papers/w7032) 和 [Avellaneda、Lee 的 PCA 统计套利框架](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf)；数据和资金费来自 [Binance Vision](https://data.binance.vision/) 与 [`/fapi/v1/fundingRate`](https://fapi.binance.com/fapi/v1/fundingRate)。这些股票市场研究不能直接作为加密货币收益证明。
