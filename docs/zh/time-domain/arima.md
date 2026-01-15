# ARIMA模型

<div class="interview-summary">
<strong>面试摘要：</strong> ARIMA(p,d,q)通过包含差分将ARMA扩展到非平稳序列。"I"代表"积分"——意味着序列需要d次差分才能变得平稳。差分d次后，对差分序列拟合ARMA(p,q)。最常见的是d=1（一阶差分）或d=2（二阶差分）。
</div>

## 核心定义

**ARIMA(p,d,q)模型**：

对$X_t$的d阶差分应用ARMA(p,q)：
$$\Phi(L)(1-L)^d X_t = c + \Theta(L)\epsilon_t$$

**成分：**
- $p$：AR阶数（自回归滞后）
- $d$：差分阶数（积分阶数）
- $q$：MA阶数（移动平均滞后）

**差分算子**：
$$\nabla X_t = (1-L)X_t = X_t - X_{t-1}$$
$$\nabla^2 X_t = X_t - 2X_{t-1} + X_{t-2}$$

**积分过程**：如果一个过程需要d次差分才能变得平稳，则称其为d阶积分过程，记作$I(d)$。

## 数学与推导

### ARIMA(0,1,0)：随机游走

$$X_t = X_{t-1} + \epsilon_t$$

或等价地：$(1-L)X_t = \epsilon_t$

一阶差分$\nabla X_t = \epsilon_t$是白噪声（平稳的）。

### 带漂移的ARIMA(0,1,0)

$$X_t = c + X_{t-1} + \epsilon_t$$

漂移$c$在水平上创建线性趋势：
$$E[X_t] = X_0 + ct$$

### ARIMA(1,1,0)：差分后的AR(1)

$$\nabla X_t = \phi \nabla X_{t-1} + \epsilon_t$$

展开：
$$(X_t - X_{t-1}) = \phi(X_{t-1} - X_{t-2}) + \epsilon_t$$
$$X_t = (1+\phi)X_{t-1} - \phi X_{t-2} + \epsilon_t$$

这是水平上有单位根的AR(2)。

### ARIMA(0,1,1)：IMA(1,1)

$$\nabla X_t = \epsilon_t + \theta\epsilon_{t-1}$$

也称为**指数加权移动平均（EWMA）**过程。构成简单指数平滑的基础。

### 一般ARIMA(p,d,q)

用滞后算子表示：
$$\Phi(L)(1-L)^d X_t = c + \Theta(L)\epsilon_t$$

其中：
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ 的根在单位圆外（平稳AR）
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$ 的根在单位圆外（可逆MA）
- $(1-L)^d$ 贡献$d$个单位根

### ARIMA预测

对于ARIMA(p,1,q)，h步预测：
$$\hat{X}_{T+h|T} = E[X_{T+h} | X_T, X_{T-1}, \ldots]$$

关键性质：对于$d \geq 1$，预测回归到线性趋势（如果有漂移）或常数增长。

**预测区间**由于累积不确定性而随预测步长增宽。

## 算法/模型概述

**Box-Jenkins方法：**

```
1. 识别
   - 绘制序列；检查趋势/非平稳性
   - 应用ADF/KPSS检验
   - 差分直到平稳（确定d）
   - 检查差分序列的ACF/PACF
   - 识别候选(p, q)阶数

2. 估计
   - 拟合候选ARIMA模型
   - 使用MLE（或CSS作为初始值）
   - 检查参数显著性

3. 诊断
   - 检查残差：ACF应无模式
   - 残差自相关的Ljung-Box检验
   - 检查残差正态性（Q-Q图）
   - 寻找异常值

4. 预测
   - 生成点预测
   - 计算预测区间
   - 如果可能，在保留数据上验证
```

**确定d：**

| 症状 | 可能的d |
|---------|----------|
| 序列漫游，ACF衰减慢 | d = 1 |
| 差分序列有趋势 | d = 2 |
| 季节模式持续 | 需要季节差分 |
| 已经围绕均值波动 | d = 0 |

## 常见陷阱

1. **过度差分**：如果原始序列是平稳的，差分会引入$\theta = -1$的MA(1)。检查：如果差分序列的ACF在滞后1处有大的负尖峰，你可能过度差分了。

2. **差分不足**：ACF不衰减或在高滞后处仍显著表明需要更多差分。也检查KPSS检验。

3. **忽略漂移**：没有常数的ARIMA(0,1,0)是纯随机游走。有漂移时有趋势。误设这个会影响长期预测。

4. **d > 2很少需要**：如果你需要d > 2，重新考虑——序列可能有其他问题（异常值、结构性断点、错误的变换）。

5. **混淆趋势类型**：确定性趋势（用回归去趋势）与随机趋势（差分）。使用错误方法会得到差结果。

6. **负预测**：对于正序列（价格、计数），ARIMA可能预测负值。考虑对数变换或受约束模型。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 生成ARIMA(1,1,1)过程
np.random.seed(42)
n = 300
phi, theta = 0.5, 0.3
eps = np.random.randn(n + 2)

# 首先生成差分序列作为ARMA(1,1)
dX = np.zeros(n)
dX[0] = eps[1] + theta * eps[0]
for t in range(1, n):
    dX[t] = phi * dX[t-1] + eps[t+1] + theta * eps[t]

# 积分得到X
X = np.cumsum(dX)

# 检验平稳性
adf_X = adfuller(X)
adf_dX = adfuller(np.diff(X))
print(f"ADF p值（水平）: {adf_X[1]:.4f}")  # 应该高（非平稳）
print(f"ADF p值（差分后）: {adf_dX[1]:.4f}")  # 应该低（平稳）

# 拟合ARIMA(1,1,1)
model = ARIMA(X, order=(1, 1, 1)).fit()
print(f"\n真实值: phi={phi}, theta={theta}")
print(f"估计值: phi={model.arparams[0]:.3f}, theta={model.maparams[0]:.3f}")

# 预测
forecast = model.forecast(steps=10)
conf_int = model.get_forecast(10).conf_int()
print(f"\n10步预测: {forecast[-1]:.2f}")
print(f"95%置信区间: [{conf_int.iloc[-1, 0]:.2f}, {conf_int.iloc[-1, 1]:.2f}]")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> ARIMA中的"I"代表什么，过程"d阶积分"是什么意思？</summary>

<div class="answer">
<strong>答案：</strong> "I"代表"积分（Integrated）"。如果一个过程恰好需要d次差分才能变得平稳，则称其为d阶积分，记作I(d)。积分是差分的逆操作——如果你对平稳序列求和（积分），你得到I(1)过程。

<strong>解释：</strong>
- I(0)：平稳（不需要差分）
- I(1)：一阶差分是平稳的（例如随机游走）
- I(2)：二阶差分是平稳的（例如水平上带漂移的随机游走）

**关键洞察：** "积分"来自连续时间的类比。在离散时间中：$X_t = \sum_{s=1}^t \epsilon_s$（积分/累加的白噪声）是I(1)。

<div class="pitfall">
<strong>常见陷阱：</strong> 混淆积分阶数和多项式次数。I(1)不是关于线性趋势——而是关于非平稳性的类型（随机与确定性）。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 如何判断序列是否被过度差分？</summary>

<div class="answer">
<strong>答案：</strong> 过度差分的迹象：
1. 差分序列的ACF在滞后1处显示大的负尖峰（通常接近-0.5）
2. 差分后方差增加（应该减少或保持相似）
3. 差分序列看起来"过度修正"，有过度交替

<strong>解释：</strong>
对平稳序列差分会添加$\theta \approx -1$的MA(1)结构：
$$(1-L)X_t = X_t - X_{t-1}$$

如果$X_t$已经平稳，差分表现得像$\epsilon_t - \epsilon_{t-1}$，这是$\theta = -1$且$\rho(1) = -0.5$的MA(1)。

**检验：** 如果$d=1$差分给出ACF(1) ≈ -0.5且所有其他ACF ≈ 0，尝试$d=0$。

<div class="pitfall">
<strong>常见陷阱：</strong> 因为"大家都这样做"而自动差分。始终先测试平稳性。许多序列（特别是收益率）已经是平稳的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 证明参数为$\theta$的ARIMA(0,1,1)产生的预测等价于$\alpha = 1/(1+\theta)$的指数平滑。</summary>

<div class="answer">
<strong>答案：</strong> ARIMA(0,1,1)：$(1-L)X_t = (1+\theta L)\epsilon_t$

最优预测可以递归地写成：
$$\hat{X}_{t+1|t} = \hat{X}_{t|t-1} + (1+\theta)^{-1}(X_t - \hat{X}_{t|t-1})$$

这正是指数平滑：$\hat{X}_{t+1} = \alpha X_t + (1-\alpha)\hat{X}_t$，其中$\alpha = \frac{1}{1+\theta}$。

<strong>推导：</strong>
从ARIMA(0,1,1)：$X_t = X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$

预测误差是：
$$e_t = X_t - \hat{X}_{t|t-1} = \epsilon_t$$

预测更新：
$$\hat{X}_{t+1|t} = X_t + \theta\hat{\epsilon}_t = X_t + \theta e_t$$

重新整理：
$$\hat{X}_{t+1|t} = X_t + \theta(X_t - \hat{X}_{t|t-1})/(1+\theta) \cdot (1+\theta)$$

令$\alpha = 1/(1+\theta)$：这给出指数平滑递推。

<div class="pitfall">
<strong>常见陷阱：</strong> 对于可逆性，需要$|\theta| < 1$，这意味着对于IMA(1,1)，$\alpha \in (0.5, 1)$。$\alpha < 0.5$的值对应不可逆的MA。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 为什么ARIMA模型的预测区间随着预测步长增加而变宽？</summary>

<div class="answer">
<strong>答案：</strong> 未来冲击$\epsilon_{T+1}, \epsilon_{T+2}, \ldots$是未知的，它们的累积效应随步长增长。对于I(d)过程，冲击有永久效应，导致方差无界增长。

<strong>随机游走（ARIMA(0,1,0)）的推导：</strong>
$$X_{T+h} = X_T + \sum_{j=1}^{h}\epsilon_{T+j}$$

预测：$\hat{X}_{T+h|T} = X_T$

误差：$X_{T+h} - \hat{X}_{T+h|T} = \sum_{j=1}^{h}\epsilon_{T+j}$

方差：$\text{Var}(X_{T+h} - \hat{X}_{T+h|T}) = h\sigma^2$

**95%预测区间：** $X_T \pm 1.96\sigma\sqrt{h}$

区间宽度以$\sqrt{h}$增长，变得任意宽。

<div class="pitfall">
<strong>常见陷阱：</strong> 期望狭窄的长期区间。ARIMA无法提供紧的长期预测——不确定性是根本的。这就是为什么判断和情景对长期规划很重要。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你有月度销售数据显示明显上升趋势。一阶差分后，ACF在滞后1、12和13处显示显著尖峰。什么模型结构可能合适？</summary>

<div class="answer">
<strong>答案：</strong> 这个模式表明SARIMA，其中：
- d=1（一阶差分处理趋势）
- 显著的滞后1表明AR(1)或MA(1)
- 显著的滞后12表明季节成分（月度数据，年度模式）
- 滞后13 = 12+1是季节和非季节的交互

**候选模型：**
- SARIMA(1,1,0)(1,0,0)[12]
- SARIMA(0,1,1)(0,1,1)[12]（航空模型）
- SARIMA(1,1,1)(1,1,0)[12]

**下一步：**
1. 应用季节差分并重新检查ACF
2. 拟合候选模型并比较AIC
3. 检查残差中的剩余模式
4. 在保留数据上验证

<div class="pitfall">
<strong>常见陷阱：</strong> 忽略季节尖峰而拟合非季节ARIMA。滞后12的自相关将持续存在于残差中，降低预测质量。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley. 第4-6章。
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第15, 17章。
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. 第9章。
4. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第5章。
