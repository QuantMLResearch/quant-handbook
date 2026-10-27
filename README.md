# 量化研究手册

**Quantitative Research Handbook**

中文 VitePress 教程，共 8 大模块、31 讲，包含理论说明与 Python/Polars/cvxpy 教学示例。第 1–31 讲均有独立源码及离线验证；合成示例运行通过不构成生产验收。

课程计划逐讲深化，面向有 Python 基础、需要系统补齐金融与数学的读者。第 1 讲已恢复至本次细化前的版本；新增学习路径与公开资源导读暂作后续修订参考。

- [学习路径与逐章掌握标准](./learning_path.md)：前置自测、31 讲深化任务与综合项目验收。
- [公开资源导读](./resources.md)：教材、大学课程、官方规则、数据及工具文档，注明指定阅读与学习产物。
- [第 1 讲](./module_01/01_ashare_quant_landscape.md)：已恢复的原有正文。

## 启动网站

```bash
npm ci
npm run dev
npm run build
npm run preview
```

## 运行 Python 示例

在项目根目录使用 Python 3.11–3.13 和 uv：

```bash
uv sync --locked
uv run --locked python -m examples.lecture_08_pit_engine_practice
npm run test:examples
```

完整回归也可以直接运行 `uv run --locked --extra ml --extra deep --extra optimization python -m unittest discover -s tests -v`。基础示例不需要额外依赖；第 19、20、22 讲使用 `--extra ml`，第 21 讲使用 `--extra deep`，第 24、25 讲使用 `--extra optimization`。依赖解析结果保存在 `uv.lock`；不要用系统环境已安装的依赖代替可复现环境验收。

| 讲次 | 模块名（加在 `python -m` 后） | 运行范围 |
| --- | --- | --- |
| 1 | `examples.lecture_01_comprehensive_cost_engine` | 合成成本敏感性计算 |
| 2 | `examples.lecture_02_backtest_sanitizer` | 合成行情中的偏差对比 |
| 3 | `examples.lecture_03_benchmark_massive` | 100 万行基准与结果一致性检查 |
| 4 | `examples.lecture_04_industrial_data_pipeline` | 默认输出用法，`--fetch` 才联网 |
| 5 | `examples.lecture_05_dynamic_universe_builder` | 动态股票池合成案例 |
| 6 | `examples.lecture_06_pit_financial_engine` | 公告时刻与更正版本案例 |
| 7 | `examples.lecture_07_industry_engine` | 行业事件与未知类别 |
| 8 | `examples.lecture_08_pit_engine_practice` | 综合收盘截面案例 |
| 9 | `examples.lecture_09_price_volume_factors` | 复权量价特征与合成性能样本 |
| 10 | `examples.lecture_10_fundamental_factors` | 同期间基本面比率与单季同比 |
| 11 | `examples.lecture_11_high_freq_to_low_freq` | OFI 三分支与按日聚合 |
| 12 | `examples.lecture_12_factor_preprocessing_pipeline` | 四步预处理与 WLS 加权正交校验 |
| 13 | `examples.lecture_13_factor_orthogonalization` | 满秩截面正交化与共线拒绝 |
| 14 | `examples.lecture_14_factor_ic_metrics` | IC 有效样本、iid/HAC 均值检验 |
| 15 | `examples.lecture_15_quantile_and_monotonicity` | 稳定分组、日收益夏普与初始净值回撤 |
| 16 | `examples.lecture_16_turnover_and_alpha_decay` | 累计/边际 IC、半衰期与显式假设情景 |
| 17 | `examples.lecture_17_fake_alpha_forensics` | 市值依赖与未成交现金实验 |
| 18 | `examples.lecture_18_financial_cv_purging_embargo` | 前向验证、双侧标签清洗与 CPCV 研究切分 |
| 19 | `examples.lecture_19_lightgbm_catboost_ranking` | 标签成熟边界与 LightGBM 排序训练 |
| 20 | `examples.lecture_20_shap_interpretability` | SHAP 输出加和校验与唯一日期 |
| 21 | `examples.lecture_21_deep_learning_boundaries` | CNN/GRU 合成张量训练，不作模型优劣比较 |
| 22 | `examples.lecture_22_ml_stock_selection_pipeline` | 独立训练/验证/校准/测试与收盘目标权重 |
| 23 | `examples.lecture_23_barra_risk_model_architecture` | 逐日历史暴露 WLS、EWMA 与特异性方差收缩 |
| 24 | `examples.lecture_24_cvxpy_portfolio_optimization` | 持有期收益与年化 TE 硬上限 |
| 25 | `examples.lecture_25_turnover_constraints_and_costs` | AUM/ADV 冲击成本与换手约束 |
| 26 | `examples.lecture_26_risk_parity_and_hrp` | EW、IVP、HRP 与统一年化风险贡献 |
| 27 | `examples.lecture_27_vectorized_vs_event_driven_backtest` | 开盘执行、T+1、收盘估值及现金/应收/股份对账 |
| 28 | `examples.lecture_28_transaction_friction_and_market_impact` | 逐笔费用、单边年换手与双边容量成本 |
| 29 | `examples.lecture_29_matching_rules_limits_suspensions` | 保守开盘撮合、流动性上限及分红送转四时点 |
| 30 | `examples.lecture_30_performance_attribution_brinson_barra` | 现金与费用齐全的 Brinson/因子归因闭合 |
| 31 | `examples.lecture_31_live_monitoring_risk_dashboard` | 相对净值、TE、分布告警、熔断滞回及显式复位 |

文章用 VitePress 的源码引用语法展示 `examples/` 中的文件，测试与网页使用同一份实现。第 5–31 讲应在项目根目录以模块方式运行，不能只复制一个文件到其他目录。`common.py` 提供契约校验，`fixtures.py` 提供小型合成数据。

联网下载样本：

```bash
uv sync --locked --extra data
uv run --locked --extra data python -m examples.lecture_04_industrial_data_pipeline --fetch
```

下载写入 `data_lake/`。股票、指数、日历接口可能受限或变更；离线回归使用注入的模拟接口，不会调用真实数据服务，也不能据此确认在线接口可用。失败不会被当成下载成功。

## 第 4–8 讲共享数据契约

时间字段使用 `pl.Date` 或 **Asia/Shanghai 本地无时区 `pl.Datetime`**。外部带时区时间先转到上海时间再去掉时区；不要直接丢掉 UTC 时区。价格与成交字段为数值，状态为布尔值。

| 表 | 必需字段 | 语义 |
| --- | --- | --- |
| 交易日历 | `trade_date` | 经核验的开市日，覆盖样本的全部上市年龄区间 |
| 股票档案 | `symbol, list_date, delist_date` | 含历史退市证券；退市日为排除边界，未退市填 Date 类型空值 |
| 行情 | `symbol, trade_date, open, high, low, close, close_hfq, volume, turnover` | OHLC 为原始价格，HFQ 单独保存；成交量为股、成交额为元 |
| 状态事件 | `symbol, effective_date, available_at, is_st, is_delisting` | 追加式状态与更正版本，必须包含已知初始状态 |
| 财报版本 | `symbol, report_period, available_at, roe, net_profit_yoy` | 保留原始披露和更正；每行类型、币种及报表范围须事先统一 |
| 行业事件 | `symbol, taxonomy, industry_name, effective_date, available_at` | 分类版本与字典显式指定，不自动映射旧版到新版 |
| 指数快照 | `symbol, asof_date, weight, available_at` | 完整批次；`weight` 为 0–1；观测时间不能倒填成来源日期 |

字段含义：

- `effective_date` 是状态开始生效的日期；`available_at` 是该条信息对策略可用的时间，两者不能混用。
- 财报 `available_at` 应综合披露时间、数据交付与处理延迟确定。只有公告日期时可采用“下一交易日开盘前可用”的保守假设，并记录假设。
- 股票池的“上市交易天数”按交易所开市日计数，包含停牌期间，不等于已下载行情行数。日历不足直接报错。
- `in_universe` 表示候选资格；它不代表现有持仓一定能够卖出。状态未知排除，行业未知单列为 `ind_未知`，不能冒充“综合”。
- PIT 引擎是收盘后查询。示例把 15:00 设为演示截止时间，并假定日线已形成；实际应计入数据延迟，信号最早下一交易时段执行。
- 非交易日或缺少应有的当日行情会抛错，不静默取最近日期。停牌需上游显式提供零量记录及估值价格。

第 4 讲补齐了样本上市日、原始/后复权价格和增量存储；它不会凭空生成 ST 历史、退市全集、财报或行业历史。第 5–8 讲用 `sample_data()` 演示完整契约，接入真实数据前需另行提供上述历史来源。

## 实现边界与回归重点

- 合成证券、财报、状态和工作日日历均为教学数据，不代表真实证券历史或交易所日历。
- 财报先过滤可用时刻，再选择每个报告期当时可见的版本；默认返回最新报告期。更正旧年报不会使当前指标退回旧期。
- 指数快照按完整批次留存，读取截止时刻前已观测的最新批次，不保留已移出的旧成分股；今日下载不能还原过去未采集的快照。
- 行情按 `(symbol, trade_date)` 合并，写入前检查有限值、OHLC 关系与唯一键；年度文件原子替换。该示例不支持多进程并发写入和跨文件事务，也不是行情修订的完整 PIT 存档。
- 测试覆盖非交易日、短行情窗口、历史状态、生效边界、财报更正、未知行业、未来数据注入和增量写盘。测试通过不构成生产验收或收益证明。
- 第 2 讲以现金和持仓核算净值，使用固定初始分仓的简化开盘交易模型；DSR 的多次试验示例明确标注试验间离散程度假设，不是策略真实概率。
- 第 3 讲只报告实测耗时比，不承诺固定加速倍数。第 8 讲按可读性组织参考实现，尚无全市场性能验收。

## 第 9–17 讲的接口与验证范围

- 默认日期列统一为 `trade_date`，评估引擎可显式配置其他列名。每日一只股票一行，重复键必须先处理；日历和停牌估值由上游负责。
- 第 9 讲跨日收益使用 `close_hfq`；第 10 讲还需要已 PIT 对齐的单季和 TTM 财务字段，前八讲的简化财报样本不会自动提供它们。同比增长不冒充 SUE。
- 第 11 讲使用上海本地无时区 `timestamp`，不跨交易日衔接 OFI；定时快照不能完整还原逐订单事件。
- 第 12 讲用 NumPy 求解 WLS，无需额外安装 statsmodels；验证的是加权正交。第 13 讲对常数与共线输入明确报错，不伪造独立因子。
- 第 14–15 讲的前瞻标签必须由调用者按实际信号/执行时点对齐；第 16 讲接收的是后向日收益，并内部前移构造标签，不能重复 shift。
- 第 15 讲是无成本的等权分组诊断。第 16 讲的成本表是参数情景，不是实际调仓回测。第 17 讲是独立买单实验，未成交权重留现金，未模拟完整连续持仓。
- `tests/test_factor_modules.py` 覆盖除权、财务期间、OFI 分支、填充和加权正交、共线、NaN 截面、并列分组、收益缺失、夏普/回撤、复利、半衰期及未成交现金。与前八讲的测试一起运行：`npm run test:examples`。

## 参考文档

- [Polars：join_asof](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html)
- [AkShare：股票接口](https://akshare.akfamily.xyz/data/stock/stock.html)
- [AkShare：指数接口](https://akshare.akfamily.xyz/data/index/index.html)
- [VitePress：引用源码](https://vitepress.dev/guide/markdown#import-code-snippets)

## 第 18–26 讲的接口与验证范围

```bash
uv sync --locked --extra ml --extra deep --extra optimization
uv run --locked --extra ml python -m examples.lecture_22_ml_stock_selection_pipeline
uv run --locked --extra optimization python -m examples.lecture_25_turnover_constraints_and_costs
npm run test:examples
```

- 日期默认 `trade_date`，使用 `pl.Date`；训练标签必须提供 `label_end_date`，固定 H 的日期推算只适用于完整交易日序列。合成工作日日历不代表真实开市日。
- 前向验证只使用过去数据；CPCV 是允许双侧训练的研究切分，不自动生成完整回测路径或 PBO。早停指标不当作最终测试业绩。
- 第 22 讲从已完成 PIT 对齐的特征面板开始，未自动接入前序财报、WLS、SHAP 或交易服务。全过滤返回空目标清单；次日涨停与未成交由执行层处理。
- `RankReturnCalibrator` 使用冻结模型的历史样本外预测与成熟标签拟合 H 日超额收益。输出 `expected_returns` 才能接入优化器，LambdaRank 原始分数不能直接当收益率。
- 风险引擎接收 `(T,N,K)` 历史期初暴露、`(T,N)` 收益/市值和独立的当前暴露。二维固定暴露需显式声明 `assume_static=True`；未复刻商业 Barra 的所有调整。
- 优化器输入收益为 H 日、协方差为年化；内部转换持有期风险。冲击成本使用同币种的 AUM/ADV 金额与日波动率；线性费率是教学假设，不代表现行税率。
- 第 24、25 讲使用 Clarabel 处理 TE 二次硬约束，求解后检查残差。第 26 讲输入日频协方差，MRC/TRC/波动率均年化；HRP 不保证 ERC。
- `tests/test_ml_risk_modules.py` 覆盖标签区间、未来标签注入、校准成熟、空组合、SHAP 加和、历史暴露、TE 上限、成本尺度和风险贡献恒等式。

本机验证发现：在已训练 LightGBM 的同一进程中执行 PyTorch 卷积可能阻塞。第 21 讲和深度模型回归测试使用独立 Python 进程运行；不要在单个长驻解释器内批量执行所有示例。


## 第 27–31 讲的接口与验证范围

```bash
uv run --locked python -m examples.lecture_27_vectorized_vs_event_driven_backtest
uv run --locked python -m examples.lecture_31_live_monitoring_risk_dashboard
npm run test:examples
```

- 第 27 讲直接调用第 28 讲成本引擎、第 29 讲撮合与公司行为账本。`trading_calendar` 为必填；每个交易日先 `execute_rebalance`，后 `close_day`，信号日期必须严格早于执行日期。
- 开盘成交不读取全日高低收盘价；可成交股数上限需显式提供，未知为零。数量规则和限价由上游证券规则表配置。未成交仓位继续计入资产，同日再次调用不会解除 T+1。
- 登记日收盘冻结分红权益，除权日确认现金应收和锁定送股，支付/上市日期分别转为可用现金/股份。账本幂等；缺少登记日快照、零碎股分配或持仓估值时显式报错。
- 费率为教学参数；容量模型的年换手为单边口径、每侧日均订单额为 `AUM × turnover / (days × stocks)`，按买卖两侧累加成本。负利润保留，输出万元按元除以 10000。
- 第 30 讲只处理期初权重对应的单期归因，现金和费用显式参与闭合。期内调仓与多期复利需切分和链接；股票残差不等于已证明的纯 Alpha。
- 第 31 讲主动收益为 `Rp - Rb`，相对净值另用基准收益还原；省略基准仅代表零收益基准。初始净值为 1，TE 默认最近 60 日，回撤历史需完整。
- 风控状态在同一实例内保留；三级不会自动解除，显式复位须满足恢复条件并记录审批引用。数据异常与严重 PSI 变化禁止新买入，`allow_new_buys` 可实际传入第 27 讲。其他降风险参数仍需接入优化器重算目标。
- `tests/test_execution_monitoring.py` 覆盖账户守恒、费用预留、同日 T+1、未来行情不影响开盘撮合、分红权益与到账、送股上市及余股清仓、容量口径、归因闭合、首日回撤、熔断滞回、常数 PSI、缺失/NaN 和执行层风控拦截。
- 示例不包含异步报撤单、队列优先级、分红税补扣、外部通知、持久化和审批鉴权。基础数据、模型、归因子期和生产执行仍需外部集成，不能据此宣称全流程已上线。
