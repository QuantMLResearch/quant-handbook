export const modules = [
  {
    "id": "01",
    "title": "工程与量化认知",
    "en": "FOUNDATIONS",
    "description": "建立研究边界，搭建现代数据与计算环境。",
    "tags": [
      "收益来源",
      "回测偏差",
      "Polars · DuckDB"
    ],
    "stage": "foundation",
    "lessons": [
      {
        "number": 1,
        "title": "第 1 讲：A 股量化生态与盈利模式全景",
        "link": "/module_01/01_ashare_quant_landscape"
      },
      {
        "number": 2,
        "title": "第 2 讲：量化回测的“四大死因”与防范心法",
        "link": "/module_01/02_backtest_pitfalls"
      },
      {
        "number": 3,
        "title": "第 3 讲：告别缓慢传统栈：Polars + DuckDB",
        "link": "/module_01/03_modern_quant_stack"
      },
      {
        "number": 4,
        "title": "第 4 讲：A 股多源数据接入与本地落库实战",
        "link": "/module_01/04_data_acquisition_pipeline"
      }
    ]
  },
  {
    "id": "02",
    "title": "数据工程与 PIT",
    "en": "POINT-IN-TIME",
    "description": "从行情到财报，构建时点一致、可追溯的数据底座。",
    "tags": [
      "动态股票池",
      "财报对齐",
      "PIT 引擎"
    ],
    "stage": "foundation",
    "lessons": [
      {
        "number": 5,
        "title": "第 5 讲：行情数据复权与动态股票池构建",
        "link": "/module_02/05_universe_filtering"
      },
      {
        "number": 6,
        "title": "第 6 讲：Point-in-Time 财报无泄漏对齐",
        "link": "/module_02/06_pit_financials"
      },
      {
        "number": 7,
        "title": "第 7 讲：行业分类演变与概念标签维护",
        "link": "/module_02/07_industry_classification"
      },
      {
        "number": 8,
        "title": "第 8 讲：【实战】手写 A 股 PIT 检索引擎",
        "link": "/module_02/08_pit_engine_practice"
      }
    ]
  },
  {
    "id": "03",
    "title": "特征工程与因子挖掘",
    "en": "FACTOR RESEARCH",
    "description": "连接经济逻辑与统计特征，完成因子构造与标准化。",
    "tags": [
      "量价与基本面",
      "微观结构",
      "中性化"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 9,
        "title": "第 9 讲：经典量价因子体系与 Polars 向量化加速",
        "link": "/module_03/09_price_volume_factors"
      },
      {
        "number": 10,
        "title": "第 10 讲：基本面与分析师预期因子挖掘",
        "link": "/module_03/10_fundamental_factors"
      },
      {
        "number": 11,
        "title": "第 11 讲：高频数据低频化特征工程（买方核心超额）",
        "link": "/module_03/11_high_freq_to_low_freq"
      },
      {
        "number": 12,
        "title": "第 12 讲：因子预处理工业级四步法",
        "link": "/module_03/12_factor_preprocessing_pipeline"
      },
      {
        "number": 13,
        "title": "第 13 讲：因子正交化与共线性消除",
        "link": "/module_03/13_factor_orthogonalization"
      }
    ]
  },
  {
    "id": "04",
    "title": "因子评估与验证",
    "en": "VALIDATION",
    "description": "检验预测力、稳定性与交易可行性，识别虚假超额。",
    "tags": [
      "IC · ICIR",
      "分层回测",
      "Alpha 衰减"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 14,
        "title": "第 14 讲：因子预测力核心指标（IC / Rank IC / ICIR）",
        "link": "/module_04/14_factor_ic_metrics"
      },
      {
        "number": 15,
        "title": "第 15 讲：分层回测与单调性检验",
        "link": "/module_04/15_quantile_and_monotonicity"
      },
      {
        "number": 16,
        "title": "第 16 讲：因子换手率与衰减曲线（Alpha Decay）",
        "link": "/module_04/16_turnover_and_alpha_decay"
      },
      {
        "number": 17,
        "title": "第 17 讲：【实战避坑】假因子的“打脸”排查案例",
        "link": "/module_04/17_fake_alpha_forensics"
      }
    ]
  },
  {
    "id": "05",
    "title": "金融机器学习",
    "en": "MACHINE LEARNING",
    "description": "从时间隔离验证到截面排序，形成可解释的选股流程。",
    "tags": [
      "Purging · Embargo",
      "LightGBM",
      "SHAP"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 18,
        "title": "第 18 讲：金融时序交叉验证（CV）的正确切法",
        "link": "/module_05/18_financial_cv_purging_embargo"
      },
      {
        "number": 19,
        "title": "第 19 讲：表格级梯度提升树（LightGBM / CatBoost）实战",
        "link": "/module_05/19_lightgbm_catboost_ranking"
      },
      {
        "number": 20,
        "title": "第 20 讲：特征重要性与可解释性（SHAP 值）",
        "link": "/module_05/20_shap_interpretability"
      },
      {
        "number": 21,
        "title": "第 21 讲：时序深度学习模型简析（LSTM / GRU / Transformer 适用边界）",
        "link": "/module_05/21_deep_learning_boundaries"
      },
      {
        "number": 22,
        "title": "第 22 讲：【实战】从多因子截面打分到全自动选股流水线",
        "link": "/module_05/22_ml_stock_selection_pipeline"
      }
    ]
  },
  {
    "id": "06",
    "title": "风险模型与组合优化",
    "en": "PORTFOLIO CONSTRUCTION",
    "description": "将预测信号转化为满足风险、成本与换手约束的组合。",
    "tags": [
      "Barra",
      "凸优化",
      "风险平价"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 23,
        "title": "第 23 讲：Barra 结构化风险模型数学解构",
        "link": "/module_06/23_barra_risk_model_architecture"
      },
      {
        "number": 24,
        "title": "第 24 讲：凸优化求解器（cvxpy）实战",
        "link": "/module_06/24_cvxpy_portfolio_optimization"
      },
      {
        "number": 25,
        "title": "第 25 讲：换手率约束与交易成本惩罚项引入",
        "link": "/module_06/25_turnover_constraints_and_costs"
      },
      {
        "number": 26,
        "title": "第 26 讲：风险平价（Risk Parity）与层级风险平价（HRP）",
        "link": "/module_06/26_risk_parity_and_hrp"
      }
    ]
  },
  {
    "id": "07",
    "title": "全真回测与交易执行",
    "en": "BACKTEST & EXECUTION",
    "description": "把交易制度、成交限制与市场冲击纳入策略检验。",
    "tags": [
      "事件驱动",
      "交易摩擦",
      "撮合规则"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 27,
        "title": "第 27 讲：A 股回测引擎架构：向量化 vs 事件驱动",
        "link": "/module_07/27_vectorized_vs_event_driven_backtest"
      },
      {
        "number": 28,
        "title": "第 28 讲：真实交易摩擦全真模拟",
        "link": "/module_07/28_transaction_friction_and_market_impact"
      },
      {
        "number": 29,
        "title": "第 29 讲：A 股特殊撮合规则代码级实现",
        "link": "/module_07/29_matching_rules_limits_suspensions"
      }
    ]
  },
  {
    "id": "08",
    "title": "绩效归因与实盘监控",
    "en": "ATTRIBUTION & MONITORING",
    "description": "拆解收益来源，建立运行监控、风险预警与复盘闭环。",
    "tags": [
      "Brinson · Barra",
      "风险预警",
      "运行监控"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 30,
        "title": "第 30 讲：业绩归因实战：赚的是 Beta 还是纯 Alpha？",
        "link": "/module_08/30_performance_attribution_brinson_barra"
      },
      {
        "number": 31,
        "title": "第 31 讲：实盘运行监控与风控预警看板",
        "link": "/module_08/31_live_monitoring_risk_dashboard"
      }
    ]
  }
] as const

export const lessonCount = modules.reduce((total, module) => total + module.lessons.length, 0)
