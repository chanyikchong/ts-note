# 季节ARIMA（SARIMA）模型

<div class="interview-summary">
<strong>面试摘要：</strong> SARIMA(p,d,q)(P,D,Q)[s]在滞后s处添加季节AR和MA项（例如，月度数据s=12）。季节差分$(1-L^s)$去除季节单位根。"航空模型"SARIMA(0,1,1)(0,1,1)[12]是季节数据的基准。识别使用常规滞后和季节滞后的ACF/PACF。
</div>

## 核心定义

**SARIMA(p,d,q)(P,D,Q)[s]模型**：
$$\Phi(L)\Phi_s(L^s)(1-L)^d(1-L^s)^D X_t = c + \Theta(L)\Theta_s(L^s)\epsilon_t$$

**成分：**
- $(p, d, q)$：非季节AR阶数、差分、MA阶数
- $(P, D, Q)$：季节AR阶数、差分、MA阶数
- $s$：季节周期（例如，月度为12，季度为4）

**多项式：**
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$（非季节AR）
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$（非季节MA）
- $\Phi_s(L^s) = 1 - \Phi_1 L^s - \cdots - \Phi_P L^{Ps}$（季节AR）
- $\Theta_s(L^s) = 1 + \Theta_1 L^s + \cdots + \Theta_Q L^{Qs}$（季节MA）

**季节差分算子**：
$$\nabla_s X_t = (1 - L^s)X_t = X_t - X_{t-s}$$

## 数学与推导

### SARIMA(0,0,0)(1,0,0)[12]：季节AR(1)

$$X_t = \Phi_1 X_{t-12} + \epsilon_t$$

ACF只在滞后12、24、36...处显著，呈指数衰减。

### SARIMA(0,0,0)(0,0,1)[12]：季节MA(1)

$$X_t = \epsilon_t + \Theta_1\epsilon_{t-12}$$

ACF只在滞后12处显著，其他地方为零。

### SARIMA(0,1,1)(0,1,1)[12]：航空模型

经典的Box-Jenkins航空乘客模型：
$$(1-L)(1-L^{12})X_t = (1+\theta L)(1+\Theta L^{12})\epsilon_t$$

展开：
$$X_t - X_{t-1} - X_{t-12} + X_{t-13} = \epsilon_t + \theta\epsilon_{t-1} + \Theta\epsilon_{t-12} + \theta\Theta\epsilon_{t-13}$$

**关键性质：**
- 一阶差分处理趋势
- 季节差分处理年度模式
- MA(1)平滑非季节噪声
- 季节MA(1)平滑年度噪声
- 滞后13处的交叉项$\theta\Theta$

### SARIMA的ACF/PACF模式

**纯季节AR (0,0,0)(P,0,0)[s]：**
- ACF：在季节滞后（s, 2s, 3s, ...）处指数衰减
- PACF：在滞后Ps处截尾

**纯季节MA (0,0,0)(0,0,Q)[s]：**
- ACF：在滞后Qs处截尾
- PACF：在季节滞后处指数衰减

**混合SARIMA：**
- 非季节模式在滞后1, 2, 3, ...
- 季节模式在滞后s, 2s, 3s, ...
- 交互模式在滞后s±1, s±2, ...

### 乘法模型结构

乘法形式意味着：
$$\Phi(L)\Phi_s(L^s) = (1-\phi_1 L)(1 - \Phi_1 L^s) = 1 - \phi_1 L - \Phi_1 L^s + \phi_1\Phi_1 L^{s+1}$$

这创建交互项（例如，对于有AR(1) × SAR(1)的月度数据，滞后13处有系数）。

## 算法/模型概述

**识别步骤：**

```
1. 绘制序列；识别季节周期s
2. 检查趋势 → 应用常规差分（d）
3. 检查季节模式 → 应用季节差分（D）
4. 通常D ≤ 1，d ≤ 2

5. 检查平稳序列的ACF/PACF：
   - 在滞后1, 2, ..., s-1：确定p, q
   - 在滞后s, 2s, 3s：确定P, Q
   - 在s±k处的尖峰：交互效应

6. 拟合候选模型
7. 比较AIC/BIC
8. 在常规和季节滞后处检查残差
```

**常见季节周期：**

| 数据频率 | 周期s |
|----------------|----------|
| 月度 | 12 |
| 季度 | 4 |
| 周（年度） | 52 |
| 日（周） | 7 |
| 小时（日） | 24 |

## 常见陷阱

1. **双季节模式**：某些数据有多个季节性（日+周）。标准SARIMA处理一个周期。考虑多季节模型或替代方法。

2. **大s导致问题**：对于s=52或s=365，估计困难。考虑傅里叶项或替代分解方法。

3. **季节差分D > 1**：很少需要，通常导致过度差分。检查D=1是否足够。

4. **忽略乘法结构**：模型是乘法的，所以当$\phi$和$\Phi$都非零时存在滞后s+1效应。

5. **非整数周期**：如果季节性不在整数滞后处（例如365.25天/年），SARIMA不直接适用。使用三角季节性。

6. **季节差分后忘记趋势**：季节差分$(1-L^{12})$不去除线性趋势。可能仍需要$d=1$。

## 简单示例

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# 生成SARIMA(1,1,1)(1,1,1)[12]数据
np.random.seed(42)
n = 200
s = 12

# 创建季节+趋势+噪声
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / s)
trend = 0.1 * t
noise = np.random.randn(n) * 2
X = trend + seasonal + np.cumsum(noise)

# 拟合SARIMA
model = SARIMAX(X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
print(results.summary().tables[1])

# 在季节滞后处检查残差
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(results.resid, lags=[12, 24], return_df=True)
print("\n季节滞后处的Ljung-Box检验:")
print(lb_test)

# 预测
forecast = results.get_forecast(steps=12)
print(f"\n12个月预测均值: {forecast.predicted_mean.values}")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 为什么SARIMA模型被称为"乘法"模型？这如何影响滞后结构？</summary>

<div class="answer">
<strong>答案：</strong> "乘法"指的是AR和MA多项式相乘：$\Phi(L) \times \Phi_s(L^s)$。这在组合滞后处创建交互项。

<strong>示例：</strong> SARIMA(1,0,0)(1,0,0)[12]：
$$(1-\phi L)(1-\Phi L^{12})X_t = \epsilon_t$$
$$X_t - \phi X_{t-1} - \Phi X_{t-12} + \phi\Phi X_{t-13} = \epsilon_t$$

滞后13处的$\phi\Phi$项是交互效应——在加法模型中不会存在。

**含义：** 对于月度数据检查滞后11、13（不仅仅是12）处的ACF/PACF。

<div class="pitfall">
<strong>常见陷阱：</strong> 期望季节和非季节效应干净分离。乘法结构混合它们，这可能使识别变得混乱。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 什么是"航空模型"，为什么它是有用的基准？</summary>

<div class="answer">
<strong>答案：</strong> 航空模型是SARIMA(0,1,1)(0,1,1)[12]，最初由Box和Jenkins拟合航空乘客数据。它有用是因为：

1. 通过一阶差分处理趋势
2. 通过季节差分处理年度季节性
3. 只使用2个参数（$\theta$, $\Theta$）却能很好地拟合许多季节序列
4. 等价于Holt-Winters指数平滑

**模型：**
$$(1-L)(1-L^{12})X_t = (1+\theta L)(1+\Theta L^{12})\epsilon_t$$

<div class="pitfall">
<strong>常见陷阱：</strong> 不检查拟合就将航空模型作为默认。对于某些数据，可能需要AR项或不同的差分。始终用残差诊断验证。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 推导季节MA(1)模型$X_t = \epsilon_t + \Theta\epsilon_{t-12}$在滞后12处的ACF。</summary>

<div class="answer">
<strong>答案：</strong> $\rho(12) = \frac{\Theta}{1+\Theta^2}$，且$\rho(h) = 0$ 对于 $h \neq 0, 12$。

<strong>推导：</strong>

**方差：**
$$\gamma(0) = \text{Var}(\epsilon_t + \Theta\epsilon_{t-12}) = \sigma^2(1 + \Theta^2)$$

**滞后12的自协方差：**
$$\gamma(12) = E[(\epsilon_t + \Theta\epsilon_{t-12})(\epsilon_{t-12} + \Theta\epsilon_{t-24})]$$
$$= E[\Theta\epsilon_{t-12}^2] = \Theta\sigma^2$$

**ACF：**
$$\rho(12) = \frac{\gamma(12)}{\gamma(0)} = \frac{\Theta\sigma^2}{\sigma^2(1+\Theta^2)} = \frac{\Theta}{1+\Theta^2}$$

对于其他滞后，没有$\epsilon$项的重叠，所以$\gamma(h) = 0$。

<div class="pitfall">
<strong>常见陷阱：</strong> 注意$|\rho(12)| \leq 0.5$，与非季节MA(1)的约束相同。观察到更大的季节相关性表明季节AR或组合模型。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 对于SARIMA(0,1,0)(0,1,0)[12]，写出模型方程并解释它代表什么。</summary>

<div class="answer">
<strong>答案：</strong> 这是"季节随机游走"：
$$(1-L)(1-L^{12})X_t = \epsilon_t$$

展开：
$$X_t - X_{t-1} - X_{t-12} + X_{t-13} = \epsilon_t$$
$$X_t = X_{t-1} + X_{t-12} - X_{t-13} + \epsilon_t$$

**解释：** 今天的值 = 昨天的值 + 去年这个月 - 去年同一天 + 噪声。

这是"朴素季节"预测：$\hat{X}_{t+1} = X_t + (X_{t+1-12} - X_{t-12})$。

它说：重复去年的季节模式，同时延续昨天的水平。

<div class="pitfall">
<strong>常见陷阱：</strong> 这个模型是有用的基准但通常过于简单。真实数据通常受益于MA项来平滑噪声。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你正在建模小时电力需求，有明显的日（24小时）和周（168小时）模式。你能使用标准SARIMA吗？有什么替代方案？</summary>

<div class="answer">
<strong>答案：</strong> 标准SARIMA只处理一个季节周期。对于多季节性，替代方案包括：

1. **双季节模型**：s=24的SARIMA加周模式的外部回归变量
2. **TBATS**：具有多个季节周期的指数平滑
3. **傅里叶项**：在两个频率处包含sin/cos项
4. **Prophet**：通过加法分解处理多季节性
5. **神经网络方法**：LSTM或Transformer可以学习复杂模式

**实际方法：**
- 使用s=24（主导模式）
- 为周模式添加星期几虚拟变量或傅里叶项
- 考虑：`SARIMAX(p,d,q)(P,D,Q)[24]`，`exog=周虚拟变量`

<div class="pitfall">
<strong>常见陷阱：</strong> 尝试使用s=168进行周季节性会导致估计问题（168太大）。改用分层或加法方法。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. 第9章。
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. 第9章。
3. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. 第3章。
4. De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *JASA*, 106(496), 1513-1527.
