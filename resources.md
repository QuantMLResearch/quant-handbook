---
title: "公开资源导读：教材、课程、数据与工具"
outline: [2, 3]
---

# 公开资源导读：教材、课程、数据与工具

这份导读用于补足主线课程的基础与延伸阅读。优先选择大学课程、作者公开教材、官方规则、原作者数据和工具文档。**公开可访问不等于可以任意转载或商用**；数据使用与软件许可应分别查看来源条款。本项目链接到原始来源，不重新分发受限教材或数据。

核验日期：**2026-10-26**。下列主表中的网页或官方入口已打开核对；网站内容和软件 `stable` 文档会更新，运行项目仍以 `uv.lock` 为准。链接可访问只说明本次可读取，不能证明数据源有完整历史或生产可用性。

## 数学与统计基础

| 编号与公开来源 | 类型、语言与门槛 | 建议读什么 | 读完做什么／衔接章节 |
| --- | --- | --- | --- |
| R01 · [MIT 18.06 Linear Algebra，Gilbert Strang](https://www.ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | 免费公开课、视频和习题；英文；高中代数基础 | 向量空间、投影与最小二乘、特征值、正定矩阵 | 用二维例子解释回归投影；判断协方差是否正定。对应第 12、13、23–26 讲 |
| R02 · [Harvard Stat 110，Joe Blitzstein](https://stat110.hsites.harvard.edu/) | 免费课程与作者在线教材入口；英文；微积分基础 | 条件概率、期望、方差、条件期望、大数定律与中心极限定理 | 说明随机变量、样本平均与总体期望的区别；判断独立性假设。对应第 2、14、18 讲 |
| R03 · [MIT 18.05 Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) | 免费课件与习题；英文；微积分基础 | 估计、置信区间、假设检验与回归 | 给一个均值估计附上标准误，解释 p 值不能说明什么。对应第 14、17、18 讲 |
| R04 · [Forecasting: Principles and Practice，第三版](https://otexts.com/fpp3/) | 作者公开在线教材；英文；代码使用 R | 第 2 章时间序列图形、第 5 章预测工具箱、第 7 章回归、第 9 章 ARIMA | 用滚动预测比较简单基线与模型，检查残差。对应第 16、18、21、31 讲；R 代码不直接作为本项目 Python 依赖 |
| R05 · [An Introduction to Statistical Learning，Python 版](https://www.statlearning.com/) 与 [作者实验材料](https://www.statlearning.com/resources-python) | 作者公开教材及 Python labs；英文；基础统计和 Python | 第 2、3 章学习框架与回归，第 5 章重采样，第 6 章正则化，第 8 章树模型 | 在普通数据上复现实验，再指出金融时序需要额外修改的验证假设。对应第 18–22 讲 |
| R06 · [Convex Optimization，Boyd 与 Vandenberghe](https://stanford.edu/~boyd/cvxbook/) | 作者获授权公开的教材、课件和示例；英文；线性代数、微积分 | 第 2 章凸集、第 3 章凸函数、第 4 章优化问题、第 5 章对偶 | 识别组合问题哪些部分是凸的，解释约束和不可行性。对应第 24、25 讲 |

学习顺序建议：R01/R02 按需补基础 → R03 理解估计 → R04/R05 学验证 → R06 学优化。不要同时从第一页读完六本；先围绕当前章的问题找对应主题。

## 金融与基准

| 编号与公开来源 | 读什么 | 学习产物与边界 |
| --- | --- | --- |
| R07 · [MIT 15.401 Finance Theory I，Andrew Lo](https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/) | 优先看 Risk and Return、Portfolio Theory、CAPM and APT；对冲部分看 Forward and Futures Contracts | 画出收益、风险、定价和对冲的关系。英文课程有视频、课件和习题；金融原理可迁移，2008 年课程不提供当前中国市场规则 |
| R08 · [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) 与 [三因子构造说明](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html) | 因子构造、组合分组、频率和历史版本说明 | 做一张因子数据字典，区分因子收益与个股暴露。该数据不是 A 股的现成风险模型；下载数据后要确认单位、市场、日期和修订 |
| R09 · [上交所股票指数计算与维护细则入口](https://www.sse.com.cn/market/sseindex/calculation/) | 股票指数文档中的价格指数、全收益指数、净收益指数、除数修正及变更记录 | 为基准记录收益口径和版本，解释现金分红如何处理；不能把指数方法文件误当成历史成分权重数据库 |
| R10 · [上交所交易规则（2026 年修订）发布页](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml) | 发布通知、规则正文及暂缓实施条文三个部分一起看 | 形成“规则名称—适用证券—生效日期—例外—代码字段”卡片。中文官方原文；此页不是深交所、北交所或中金所的通用规则 |

**第 1 讲最低阅读组合**：R07 的 Risk and Return + Forward and Futures Contracts，R09 的三种指数口径，以及 R10 的生效与例外检查。R08 可在读懂回归后补读。

中金所的 [IM 合约文本](https://www.cffex.com.cn/u/cms/www/202207/18190308vxpv.pdf) 可作为进一步查阅入口。本次搜索能定位官方合约内容，但直接读取出现超时，故不将其计入主表的可直接读取资源；不要据此认定任何实时保证金或交易限额。第 1 讲的数值实验使用明确给定的合约参数假设。

## 数据、研究工具与工程

| 编号与官方来源 | 建议阅读与实验 | 对应章节与注意点 |
| --- | --- | --- |
| R11 · [uv 官方文档](https://docs.astral.sh/uv/) | Projects、Locking and syncing；在新环境按锁文件重建项目 | 第 3 讲；区分项目依赖、解释器版本和锁文件 |
| R12 · [Polars User Guide](https://docs.pola.rs/user-guide/) | Expressions、Lazy API、Joins、Window functions | 第 3–13 讲；先检查排序、键和空值，再比较性能 |
| R13 · [DuckDB 官方文档](https://duckdb.org/docs/stable/) | SQL、Parquet 导入、查询与执行计划 | 第 3、4、8 讲；用同一小样本与 DataFrame 结果对照 |
| R14 · [AKShare 股票接口文档](https://akshare.akfamily.xyz/data/stock/stock.html) | 每个实际使用接口的来源、频率、参数、单位和调整选项 | 第 4–8 讲；接口开源不代表底层数据没有许可限制，也不保证历史数据完整 |
| R15 · [scikit-learn：Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | 先划分数据，再拟合变换；阅读泄漏示例 | 第 2、12、18–22 讲；Pipeline 不会自动替你确定金融标签时间边界 |
| R16 · [statsmodels：Linear Regression](https://www.statsmodels.org/stable/regression.html) | OLS、WLS、GLS 与估计假设 | 第 12、14、23、30 讲；可对照本项目 NumPy 实现，非当前必须安装依赖 |
| R17 · [LightGBM 官方文档](https://lightgbm.readthedocs.io/en/stable/) | Python Quick Start、Parameters、Parameters Tuning | 第 19、22 讲；区分排序分数与收益率，记录训练与验证版本 |
| R18 · [CVXPY：What is CVXPY?](https://www.cvxpy.org/tutorial/intro/index.html) | 变量、约束、问题状态、向量矩阵，再读 DCP | 第 24、25 讲；复现一个不可行问题并检查状态，不只读取 `w.value` |
| R19 · [Microsoft Qlib 官方仓库](https://github.com/microsoft/qlib) | README 中的数据、模型、工作流和评估入口 | 第 22、27 讲后作为框架对照；说明其假设与本项目接口的差异。开源框架的存在不证明默认配置已符合你的执行规则 |

这些资源用于理解与对照，不要求一次安装全部工具。公开项目应固定 tag/commit、依赖、数据快照和随机种子后再复现；“最新版能运行”不是可重复的实验定义。

## 怎样阅读一篇量化论文或一套开源实现

先写一张阅读卡，再决定是否复现：

1. **问题与机制**：预测、解释、定价还是执行？经济机制是什么？
2. **样本与信息时点**：市场、证券范围、期间、退出样本、特征可用时刻是什么？
3. **对象与单位**：预测值是排序分数还是收益率？损益是金额、百分比还是风险调整后的量？
4. **对照与选择**：用了哪些简单基线？调参、特征选择和试验次数如何处理？
5. **成本与可实现性**：卖空、换手、冲击、交易规则和容量假设是什么？
6. **复现差异**：原数据能否获得？替代数据会改变什么结论？
7. **证据强度**：合成演示、样本内拟合、样本外验证和生产结果分别支持什么？

论文、代码和复现报告是不同证据层。后续逐章将增加对应的原始论文，并明确版本与实验条件；不把尚未阅读或尚未运行的材料标为“已复现”。

## 资源维护约定

新增材料应记录标题、作者/机构、版本或日期、公开入口、语言、前置知识、指定阅读、验收产物和访问限制。优先引用原作者、大学、交易所和项目官方文档，避免只给转载文章或搜索结果页。

交易规则按生效日期维护；教材按版次维护；软件按项目锁定版本维护；数据按快照和许可证维护。外部资源更新后，本课程应重新核对受影响的例子和结论，而不只替换链接。
