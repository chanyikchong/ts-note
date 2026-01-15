# 时间序列学习笔记

<div class="interview-summary">
<strong>欢迎来到时间序列学习笔记</strong> — 一个全面的双语（English/中文）资源，用于学习时间序列分析和预测，专为面试准备和实际应用设计。
</div>

## 内容概览

本知识库涵盖：

- **基础概念**：平稳性、自相关、偏自相关
- **时域模型**：AR、MA、ARMA、ARIMA、SARIMA的识别和估计
- **指数平滑**：SES、Holt、Holt-Winters、ETS框架
- **分解方法**：STL、经典分解、季节性处理
- **预测**：预测区间、多步预测策略、滚动评估
- **模型选择**：AIC/BIC、时间序列交叉验证、残差诊断
- **谱分析**：周期图、频域基础
- **状态空间模型**：卡尔曼滤波、局部水平和趋势模型
- **多元时间序列**：VAR、VARMA、格兰杰因果
- **变点检测**：变点和异常检测方法
- **特征工程**：经典流程、缩放、缺失数据处理
- **深度学习**：RNN/LSTM/TCN、Transformer用于时间序列
- **实践建模**：回测、部署、常见陷阱

## 页面结构

每个主题页面遵循一致的8部分格式：

1. **面试摘要** — 3-6行的关键要点
2. **核心定义** — 基本术语和概念
3. **数学与推导** — 严格的数学基础
4. **算法/模型概述** — 方法如何工作
5. **常见陷阱** — 需要避免的错误
6. **简单示例** — 快速说明
7. **测验** — 5+道题目，答案默认隐藏（点击展开）
8. **参考文献** — 延伸阅读

## 开始学习

从侧边栏选择一个主题开始。每个页面都是独立的，但建立在基础概念之上。

**初学者推荐学习路径：**

1. 从[平稳性](foundations/stationarity.md)和[自相关](foundations/autocorrelation.md)开始
2. 进入[AR模型](time-domain/ar.md) → [MA模型](time-domain/ma.md) → [ARMA](time-domain/arma.md) → [ARIMA](time-domain/arima.md)
3. 学习[模型识别](time-domain/identification.md)和[残差诊断](model-selection/residual-diagnostics.md)
4. 探索[指数平滑](exponential-smoothing/ses.md)和[分解](decomposition/stl.md)
5. 深入[状态空间模型](state-space/kalman-filter.md)和[多元时间序列](multivariate/var.md)

## 代码示例

可运行的Python演示在`ts_examples/`目录中。运行方式：

```bash
python -m ts_examples.run --demo <demo_name>
```

可用演示：`arima`、`ets`、`stl`、`kalman`、`var`、`changepoint`、`backtest`、`metrics`

## 语言切换

使用页面顶部的语言选择器在English和中文之间切换。网站在两种语言中保持平行内容。

---

*这是一个开放、可扩展的知识库。请参阅仓库README了解如何添加新内容。*
