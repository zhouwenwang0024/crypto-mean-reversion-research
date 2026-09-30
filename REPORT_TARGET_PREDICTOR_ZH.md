# 逐目标 peer 公允价差研究

## 当前结论

20 个目标币都用其余 19 个币的历史对数价格做了逐月形成。形成只用过去 60 天小时数据，紧邻的 28 天只做 OOS 校准；实际信号用同一根 5 分钟收盘计算目标币与冻结 peer basket 的残差，下一根开盘成交。这是同刻 peer-implied fair value residual，不是把未来价格当作已知。

原先的 OLS/Ridge 形成共 1,080 个模型月份，经过 R²、ADF、半衰期、尺度稳定性和每月×模型 20 个目标的 BH 校正后只剩 14 个候选月份。大部分币和月份被放弃，避免把共同市场趋势造成的价格 R² 当成预测能力。

单目标账户的 5 bp 结果只能作为机会诊断，不能把各目标收益相加当作组合收益。严格的目标币加冻结 peer basket 双腿回测显示：OLS 有 7 个目标产生 32 笔交易，收益简单平均 +0.366%；Ridge 有 3 个目标产生 9 笔交易，简单平均 +0.151%。收益集中在少数币和月份，尚不足以称为稳健盈利。

本轮新增了 5 种系数和为 1 的中性 peer 模型：equal19、simplex ridge、signed neutral ridge、PCA3 neutral、sparse3 neutral。共 2,700 个模型月份诊断，31 个通过 OOS 筛选；这只是形成资格，不代表收益筛选。随后把同一模型的所有目标放入一个账户，最多 3 仓、每仓使用当前权益 30%，按同一时刻同一币净订单收费，并独立重放库存。28 天持仓、|z|=3 入场、|z|=0.5 退出、每小时观察、5 bp 含资金费的结果如下：equal19 +2.822%、neutral_ridge +2.363%、PCA3 neutral +1.365%、simplex_ridge +3.755%、sparse3 neutral -0.756%。56 天持仓对应 +2.822%、+2.263%、+1.365%、+3.490%、-0.604%。这说明在这段已知历史上存在正收益候选，但收益集中于少数币和日期，不能称为未见样本外验证。

费用敏感性使用固定信号和真实净成交额重跑。28 天版本在 0/2/5/10 bp（均含资金费）下分别为：equal19 2.970/2.911/2.822/2.674%，neutral_ridge 2.811/2.632/2.363/1.915%，PCA3 neutral 1.457/1.420/1.365/1.274%，simplex_ridge 4.296/4.080/3.755/3.213%，sparse3 neutral -0.523/-0.616/-0.756/-0.990%。所有组合的独立库存重放误差小于 3e-15，最大毛敞口小于 0.732。2025-10-11 UTC 和 Asia/Shanghai 两个事件窗口均无跨越交易，事件区间收益置零与完整结果一致；这只能说明该组合结果没有靠这一天的跨越交易支撑。

28 天组合的回撤、交易数和最大毛敞口为：

| 模型 | 收益 | 最大回撤 | 交易数 | 最大毛敞口 |
|---|---:|---:|---:|---:|
| equal19 | 2.822% | -1.439% | 5 | 0.342 |
| neutral_ridge | 2.363% | -2.320% | 15 | 0.694 |
| PCA3 neutral | 1.365% | -0.785% | 3 | 0.342 |
| simplex_ridge | 3.755% | -2.866% | 18 | 0.707 |
| sparse3 neutral | -0.756% | -3.294% | 8 | 0.731 |

## 交易和审计口径

- 目标腿和 peer basket 同时建立，peer 系数和为 1，避免把共同市场方向误当成套利。
- 每个新仓预算当前权益 30%，最多 3 个仓；所有腿都计入手续费和资金费。
- 信号使用已完成 5 分钟收盘，成交使用下一根开盘；月界取消旧月待成交信号并平仓旧 lot。
- 退出包括残差回到退出带、最长持仓和月界；同一目标回到 1σ 内才允许重新入场。
- 独立 signed-inventory replay、未来后缀扰动、反向方向和月界平仓测试必须通过。

2025-10-11 不能直接被当作盈利来源。UTC 窗口的目标预测回测没有跨越交易；Ridge 的 TRX 在 Asia/Shanghai 窗口有 1 笔跨越交易，事件区间权益变化约 +0.0449%。因此“事件区间收益置零”只是一项归因诊断，不能冒充删除整笔跨越交易后的验证收益。后续组合结果同时报告事件置零和跨越交易清除两种口径。

## 文件

- 形成：[target_predictor.py](src/extended_data/target_predictor.py)、[run_target_predictor.py](src/extended_data/run_target_predictor.py)
- 中性形成：[neutral_peer_models.py](src/extended_data/neutral_peer_models.py)、[run_neutral_peer_formation.py](src/extended_data/run_neutral_peer_formation.py)
- 目标双腿回测：[run_target_predictor_backtest.py](src/extended_data/run_target_predictor_backtest.py)
- 中性候选：[neutral_peer_selected.csv](results/neutral_peer_selected.csv)、[neutral_peer_summary.csv](results/neutral_peer_summary.csv)
- 旧目标候选：[target_predictor_selected.csv](results/target_predictor_selected.csv)
- 形成诊断：results/neutral_peer_diagnostics.csv.gz
- 组合回测：[run_neutral_peer_backtest.py](src/extended_data/run_neutral_peer_backtest.py)
- 组合结果： [neutral_peer_portfolio_summary_h28_e3_o12_net.csv](results/neutral_peer_portfolio_summary_h28_e3_o12_net.csv)、[neutral_peer_portfolio_costs_h28_e3_o12_net.csv](results/neutral_peer_portfolio_costs_h28_e3_o12_net.csv)、[neutral_peer_portfolio_events_h28_e3_o12_net.csv](results/neutral_peer_portfolio_events_h28_e3_o12_net.csv)、[neutral_peer_portfolio_manifest_h28_e3_o12_net.json](results/neutral_peer_portfolio_manifest_h28_e3_o12_net.json)、[neutral_peer_portfolio_summary_hold56_entry3_exit05_obs12.csv](results/neutral_peer_portfolio_summary_hold56_entry3_exit05_obs12.csv)
- 机制候选：[pair_mechanism_study.py](src/extended_data/pair_mechanism_study.py)、[pair_mechanism_summary.csv](results/pair_mechanism_summary.csv)
- 测试：[test_neutral_peer_models.py](tests/test_neutral_peer_models.py)、[test_neutral_peer_backtest.py](tests/test_neutral_peer_backtest.py)、[test_target_predictor_basket.py](tests/test_target_predictor_basket.py)

目前最好的已知历史候选仍是 Johansen 56 天跨月版本，5 bp 含资金费约 +4.90%，但开发期为负、扩展期仅约 +0.59%，稳定性过滤后收益明显下降；它应继续作为已知样本鲁棒性候选，而不是已验证策略。
