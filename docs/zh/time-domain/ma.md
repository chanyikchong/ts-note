# 移动平均（MA）模型

<div class="interview-summary">
<strong>面试摘要：</strong> MA(q)模型将当前值表示为当前和过去q个噪声项的线性组合。MA过程总是平稳的（白噪声的有限线性组合）。ACF在滞后q处截尾；PACF呈指数衰减。估计需要非线性优化（MLE）。可逆性要求根在单位圆外。
</div>

## 核心定义

**MA(q)模型**：
$$X_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q}$$

其中$\epsilon_t \sim WN(0, \sigma^2)$。

**滞后算子形式**：
$$X_t = \mu + \Theta(L)\epsilon_t$$

其中$\Theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q$。

**特征多项式**：
$$\Theta(z) = 1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q$$

**可逆性条件**：$\Theta(z) = 0$的所有根必须在单位圆外。

## 数学与推导

### MA(1)模型：$X_t = \mu + \epsilon_t + \theta\epsilon_{t-1}$

**均值**：$E[X_t] = \mu$

**方差**：
$$\gamma(0) = \text{Var}(X_t) = \sigma^2(1 + \theta^2)$$

**滞后1的自协方差**：
$$\gamma(1) = E[(\epsilon_t + \theta\epsilon_{t-1})(\epsilon_{t+1} + \theta\epsilon_t)] = \theta\sigma^2$$

**滞后$h \geq 2$的自协方差**：$\gamma(h) = 0$

**ACF**：
$$\rho(1) = \frac{\theta}{1+\theta^2}, \quad \rho(h) = 0 \text{ 当 } h \geq 2$$

**注意**：在$\theta = \pm 1$时最大$|\rho(1)| = 0.5$。

### MA(q)一般ACF

$$\gamma(h) = \begin{cases} \sigma^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h} & 0 \leq h \leq q \\ 0 & h > q \end{cases}$$

其中$\theta_0 = 1$。

$$\rho(h) = \frac{\sum_{j=0}^{q-h} \theta_j \theta_{j+h}}{\sum_{j=0}^{q} \theta_j^2}$$

### 可逆性和AR(∞)表示

可逆的MA(1)可以写成AR(∞)：
$$X_t = \mu + \epsilon_t + \theta\epsilon_{t-1}$$

如果$|\theta| < 1$：
$$\epsilon_t = \sum_{j=0}^{\infty}(-\theta)^j(X_{t-j} - \mu)$$

这给出：
$$X_t = \mu(1+\theta) - \theta X_{t-1} + \theta^2 X_{t-2} - \theta^3 X_{t-3} + \cdots + \epsilon_t$$

**为什么可逆性重要**：允许用可观测量表示冲击。预测和模型解释所必需。

### MA(1)的PACF

MA(1)的PACF呈指数衰减：
$$\phi_{hh} = \frac{-(-\theta)^h(1-\theta^2)}{1-\theta^{2(h+1)}}$$

对于大的$h$：$\phi_{hh} \approx -(-\theta)^h$

## 算法/模型概述

**估计方法：**

1. **创新算法**：从自协方差递归计算MA系数的方法。

2. **条件平方和（CSS）**：
   - 将样本前的$\epsilon$值设为零
   - 最小化$\sum \epsilon_t^2$
   - 快但可能有偏

3. **精确最大似然（MLE）**：
   - 考虑初始条件
   - 使用卡尔曼滤波或直接似然
   - 渐近最有效

**估计挑战：**
- MA估计是非线性的（不像AR）
- 可能存在多个局部最优
- 需要好的初始值
- 必须强制可逆性约束

**阶数选择：**
```
1. 检查ACF - 截尾表明MA阶数
2. 如果ACF在滞后q后截尾，从MA(q)开始
3. 拟合候选模型
4. 比较AIC/BIC
5. 检查残差ACF/PACF
```

## 常见陷阱

1. **参数识别**：MA(1)与参数$\theta$和MA(1)与参数$1/\theta$给出相同的ACF！始终强制可逆性以获得唯一解。

2. **估计困难**：MA模型比AR更难估计。差的初始值导致收敛问题。使用method="innovations"或CSS获取初始估计。

3. **混淆MA阶数和差分**：差分后滞后1处的大负尖峰通常表明过度差分，而非MA(1)。

4. **误解ACF截尾**："截尾"意味着突然降至零，而不仅仅是衰减。AR过程也显示ACF模式——检查PACF以区分。

5. **不可逆估计**：如果估计$|\theta| > 1$，模型不可逆。要么翻转为$1/\theta$，要么重新考虑模型设定。

6. **忽略单位根边界**：$\theta = -1$或$\theta = 1$是不可逆的。在这些值附近，标准推断失效。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

# 生成MA(2)过程
np.random.seed(42)
n = 300
theta1, theta2 = 0.6, 0.3
eps = np.random.randn(n + 2)
X = np.zeros(n)

for t in range(n):
    X[t] = eps[t+2] + theta1*eps[t+1] + theta2*eps[t]

# 检查ACF（应在滞后2后截尾）
acf_values = acf(X, nlags=10)
print("ACF:", np.round(acf_values, 3))
# 预期：滞后1、2显著；之后接近零

# 拟合MA(2)模型
model = ARIMA(X, order=(0, 0, 2)).fit()
print(f"真实值: theta1={theta1}, theta2={theta2}")
print(f"估计值: theta1={model.maparams[0]:.3f}, theta2={model.maparams[1]:.3f}")

# 理论ACF对比
gamma0 = 1 + theta1**2 + theta2**2
rho1 = (theta1 + theta1*theta2) / gamma0
rho2 = theta2 / gamma0
print(f"理论 rho(1)={rho1:.3f}, rho(2)={rho2:.3f}")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 为什么MA过程无论参数值如何总是平稳的？</summary>

<div class="answer">
<strong>答案：</strong> MA(q)是白噪声的有限线性组合：$X_t = \mu + \sum_{j=0}^{q}\theta_j\epsilon_{t-j}$。均值是常数（$\mu$），方差是$\sigma^2\sum\theta_j^2$（常数），自协方差只依赖于滞后（不依赖于时间）。

<strong>解释：</strong>
平稳性要求：
1. 常数均值：$E[X_t] = \mu$ ✓
2. 常数方差：$\text{Var}(X_t) = \sigma^2(1+\theta_1^2+\cdots+\theta_q^2)$ ✓
3. 自协方差只依赖于滞后：$\gamma(h)$不依赖于$t$ ✓

对于任何有限的$\theta$值所有条件都满足，因为白噪声是平稳的，有限线性组合保持平稳性。

<div class="pitfall">
<strong>常见陷阱：</strong> 混淆平稳性和可逆性。MA总是平稳的但不总是可逆的。可逆性是关于AR(∞)表示的，不是平稳性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 解释可逆性的概念。为什么我们关心它？</summary>

<div class="answer">
<strong>答案：</strong> 可逆性意味着我们可以将不可观测的冲击$\epsilon_t$表示为过去可观测量$X_t, X_{t-1}, \ldots$的收敛函数。我们关心是因为：
1. 它确保唯一的模型识别
2. 它使计算残差进行诊断成为可能
3. 它是正确预测更新所需的

<strong>技术细节：</strong> 对于MA(1)：$X_t = \epsilon_t + \theta\epsilon_{t-1}$

如果$|\theta| < 1$：$\epsilon_t = X_t - \theta X_{t-1} + \theta^2 X_{t-2} - \cdots$（收敛）
如果$|\theta| > 1$：展开发散

**关键方程：** MA(q)可逆当且仅当$\Theta(z) = 0$的所有根在单位圆外。

<div class="pitfall">
<strong>常见陷阱：</strong> 参数为$\theta$和$1/\theta$的模型产生相同的ACF但不同的预测。不强制可逆性，你可能得到表现不好的"错误"模型。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 证明对于MA(1)，ACF满足$|\rho(1)| \leq 0.5$。</summary>

<div class="answer">
<strong>答案：</strong> 我们有$\rho(1) = \frac{\theta}{1+\theta^2}$。取导数并令其为零找最大值。

<strong>推导：</strong>
$$\frac{d\rho(1)}{d\theta} = \frac{(1+\theta^2) - \theta(2\theta)}{(1+\theta^2)^2} = \frac{1-\theta^2}{(1+\theta^2)^2}$$

令其等于零：$\theta = \pm 1$

在$\theta = 1$时：$\rho(1) = \frac{1}{1+1} = 0.5$
在$\theta = -1$时：$\rho(1) = \frac{-1}{1+1} = -0.5$

当$\theta \to 0$时：$\rho(1) \to 0$
当$|\theta| \to \infty$时：$\rho(1) \to 0$

因此$|\rho(1)| \leq 0.5$，等号在$\theta = \pm 1$时取得。

<div class="pitfall">
<strong>常见陷阱：</strong> 如果你观察到样本$|\hat{\rho}(1)| > 0.5$，不太可能是纯MA(1)。考虑AR(1)（可以有任何$|\rho(1)| < 1$）或混合ARMA。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 推导MA(2)的方差$\gamma(0)$：$X_t = \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2}$。</summary>

<div class="answer">
<strong>答案：</strong> $\gamma(0) = \sigma^2(1 + \theta_1^2 + \theta_2^2)$

<strong>推导：</strong>
$$\gamma(0) = \text{Var}(X_t) = \text{Var}(\epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2})$$

由于$\epsilon_t$、$\epsilon_{t-1}$、$\epsilon_{t-2}$独立：
$$= \text{Var}(\epsilon_t) + \theta_1^2\text{Var}(\epsilon_{t-1}) + \theta_2^2\text{Var}(\epsilon_{t-2})$$
$$= \sigma^2 + \theta_1^2\sigma^2 + \theta_2^2\sigma^2$$
$$= \sigma^2(1 + \theta_1^2 + \theta_2^2)$$

**MA(q)的一般公式：**
$$\gamma(0) = \sigma^2\sum_{j=0}^{q}\theta_j^2 \text{ 其中 } \theta_0 = 1$$

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记按惯例$\theta_0 = 1$。公式中的"1"来自$\epsilon_t$项（系数1）。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你拟合了MA(1)模型，得到$\hat{\theta} = 1.2$。你应该怎么做？</summary>

<div class="answer">
<strong>答案：</strong> 估计$\hat{\theta} = 1.2$在可逆区域之外。选项：
1. 翻转为可逆形式：使用$\theta' = 1/1.2 = 0.833$
2. 强制可逆性约束重新拟合
3. 重新考虑模型设定（也许ARMA更好）

<strong>解释：</strong>
$\theta = 1.2$的MA(1)和$\theta = 0.833$的MA(1)产生相同的ACF：
- $\rho(1) = \frac{1.2}{1+1.44} = \frac{0.833}{1+0.694} = 0.492$

但只有$\theta = 0.833$是可逆的。计算预测或残差时需要可逆形式。

**行动计划：**
1. 检查软件是否自动强制可逆性
2. 如果没有，手动转换：$\theta_{new} = 1/\hat{\theta}$
3. 调整方差估计：$\sigma^2_{new} = \hat{\sigma}^2 \cdot \hat{\theta}^2$

<div class="pitfall">
<strong>常见陷阱：</strong> 忽略不可逆性警告。不可逆模型会给出差的预测，因为AR(∞)展开发散。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. 第4章。
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第4章。
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第3章。
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. 第3章。
