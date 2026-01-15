# ARMA模型

<div class="interview-summary">
<strong>面试摘要：</strong> ARMA(p,q)结合AR和MA成分：$\Phi(L)X_t = \Theta(L)\epsilon_t$。ACF和PACF都拖尾（衰减）。平稳性取决于AR部分；可逆性取决于MA部分。通过MLE估计。当两种模式都存在时，比纯AR或MA更简洁。
</div>

## 核心定义

**ARMA(p,q)模型**：
$$X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + \cdots + \theta_q\epsilon_{t-q}$$

**滞后算子形式**：
$$\Phi(L)X_t = c + \Theta(L)\epsilon_t$$

其中：
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$（AR多项式）
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$（MA多项式）

**均值**：$\mu = \frac{c}{1 - \phi_1 - \cdots - \phi_p}$

**平稳性**：$\Phi(z) = 0$的根在单位圆外

**可逆性**：$\Theta(z) = 0$的根在单位圆外

## 数学与推导

### ARMA(1,1)：$X_t = c + \phi X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$

**平稳性**：$|\phi| < 1$

**可逆性**：$|\theta| < 1$

**均值**：$\mu = \frac{c}{1-\phi}$

**方差**（假设$\mu = 0$简化）：
$$\gamma(0) = \phi\gamma(1) + \sigma^2(1 + \theta\phi + \theta^2)$$

结合$\gamma(1) = \phi\gamma(0) + \theta\sigma^2$求解：
$$\gamma(0) = \sigma^2 \frac{1 + 2\theta\phi + \theta^2}{1-\phi^2}$$

**ACF**：
$$\rho(1) = \frac{(1+\theta\phi)(\phi+\theta)}{1 + 2\theta\phi + \theta^2}$$
$$\rho(h) = \phi\rho(h-1) \text{ 当 } h \geq 2$$

注意：滞后1之后ACF像AR(1)衰减，但$\rho(1)$与AR(1)不同。

### 一般ARMA(p,q)的ACF

对于$h > q$：
$$\gamma(h) = \phi_1\gamma(h-1) + \phi_2\gamma(h-2) + \cdots + \phi_p\gamma(h-p)$$

超出q的滞后，ACF满足与AR(p)相同的递推关系。初始值$\gamma(0), \ldots, \gamma(q)$同时依赖AR和MA参数。

### 因果和可逆表示

**因果（MA(∞)）形式**：如果平稳：
$$X_t = \mu + \sum_{j=0}^{\infty}\psi_j\epsilon_{t-j}$$

其中$\psi_j$系数来自$\Psi(L) = \Theta(L)/\Phi(L)$。

**可逆（AR(∞)）形式**：如果可逆：
$$\Pi(L)(X_t - \mu) = \epsilon_t$$

其中$\Pi(L) = \Phi(L)/\Theta(L)$。

### 参数冗余

**关键**：确保AR和MA多项式没有公共根（因子）。

例如：$X_t = 0.5X_{t-1} + \epsilon_t - 0.5\epsilon_{t-1}$

这里$(1-0.5L)X_t = (1-0.5L)\epsilon_t$，简化为$X_t = \epsilon_t$（白噪声！）。

这称为**公因子**或**参数冗余**。

## 算法/模型概述

**识别：**
```
1. 检查平稳性；如需要则差分
2. 检查ACF和PACF：
   - 两者都拖尾 → ARMA（不是纯AR或MA）
   - ACF截尾 → 可能MA
   - PACF截尾 → 可能AR
3. 使用EACF（扩展ACF）或信息准则
4. 拟合候选模型
5. 比较AIC/BIC，检查残差
```

**扩展ACF（EACF）方法：**

EACF通过迭代移除AR结构简化识别。结果表格显示指示(p,q)阶数的"O"模式。

**估计：**

1. **条件MLE**：条件于初始值，最大化似然
2. **精确MLE**：正确考虑初始条件
3. **CSS（条件平方和）**：最小化残差平方和

大多数软件默认使用精确MLE。

## 常见陷阱

1. **过度参数化**：ARMA(2,2)通常不比ARMA(1,1)好。简洁性对预测很重要。

2. **公因子问题**：如果多项式共享一个根，ARMA(p,q)可能简化为ARMA(p-1,q-1)。检查参数冗余。

3. **局部最优**：ARMA似然可能有多个极值。尝试不同的初始值。

4. **近似抵消**：参数接近抵消（例如$\phi \approx \theta$）导致估计不稳定和标准误膨胀。

5. **识别混淆**：ACF和PACF都拖尾，但模式不同。关注整体衰减率并与理论模式比较。

6. **忘记条件**：需要平稳性（AR根）和可逆性（MA根）都在单位圆外。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

# 生成ARMA(1,1)过程
np.random.seed(42)
n = 500
phi, theta = 0.7, 0.4
eps = np.random.randn(n + 1)
X = np.zeros(n)

X[0] = eps[1] + theta * eps[0]
for t in range(1, n):
    X[t] = phi * X[t-1] + eps[t+1] + theta * eps[t]

# ACF和PACF都应拖尾
print("ACF（前6个）:", np.round(acf(X, nlags=5), 3))
print("PACF（前6个）:", np.round(pacf(X, nlags=5), 3))

# 拟合ARMA(1,1)
model = ARIMA(X, order=(1, 0, 1)).fit()
print(f"\n真实值: phi={phi}, theta={theta}")
print(f"估计值: phi={model.arparams[0]:.3f}, theta={model.maparams[0]:.3f}")

# 检查参数冗余
print(f"\nPhi - Theta = {abs(model.arparams[0] - model.maparams[0]):.3f}")
# 如果接近0，可能近似抵消

# 与纯AR和纯MA比较
ar_aic = ARIMA(X, order=(2, 0, 0)).fit().aic
ma_aic = ARIMA(X, order=(0, 0, 2)).fit().aic
arma_aic = model.aic
print(f"\nAIC: AR(2)={ar_aic:.1f}, MA(2)={ma_aic:.1f}, ARMA(1,1)={arma_aic:.1f}")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 为什么ARMA模型可以比纯AR或MA模型更简洁？</summary>

<div class="answer">
<strong>答案：</strong> 许多实际过程既有自回归动态（动量/持续性）又有随时间消散的冲击效应。用纯AR或MA建模需要许多参数，而ARMA用更少的参数捕获两者。

<strong>示例：</strong> 需要AR(10)或MA(10)的过程可能用只有2个参数的ARMA(1,1)很好地近似。

**关键洞察：** ARMA(1,1)有无限ACF衰减（像AR(∞)）和无限PACF衰减（像MA(∞)），简洁地实现复杂相关结构。

<div class="pitfall">
<strong>常见陷阱：</strong> 假设更多参数更好。ARMA(3,3)经常过拟合。从简单开始——ARMA(1,1)通常就足够了。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> ARMA模型中的参数冗余是什么，为什么有问题？</summary>

<div class="answer">
<strong>答案：</strong> 参数冗余（公因子问题）发生在AR和MA多项式共享一个根时，导致它们相消。模型简化为低阶ARMA。

<strong>示例：</strong>
$$(1 - 0.5L)X_t = (1 - 0.5L)\epsilon_t$$

两边都有因子$(1-0.5L)$。相消得到$X_t = \epsilon_t$。

**问题：**
1. 额外参数不改善拟合
2. 估计变得不稳定（近奇异Hessian）
3. 标准误爆炸
4. 误导性的模型复杂度

**检测：** 检查AR和MA根是否接近。大标准误表明近似冗余。

<div class="pitfall">
<strong>常见陷阱：</strong> 不检查公因子。软件可能在ARMA(1,1)足够时拟合ARMA(2,2)，导致不稳定估计。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 对于ARMA(1,1)，证明ACF满足$\rho(h) = \phi^{h-1}\rho(1)$ 当 $h \geq 1$。</summary>

<div class="answer">
<strong>答案：</strong> 对于$h \geq 2$，自协方差满足AR(1)递推：$\gamma(h) = \phi\gamma(h-1)$。

<strong>推导：</strong>
将$X_t = \phi X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$两边乘以$X_{t-h}$并取期望：

对于$h \geq 2$：
$$E[X_t X_{t-h}] = \phi E[X_{t-1}X_{t-h}] + E[\epsilon_t X_{t-h}] + \theta E[\epsilon_{t-1}X_{t-h}]$$

由于当$h \geq 2$时$\epsilon_t$和$\epsilon_{t-1}$与$X_{t-h}$不相关：
$$\gamma(h) = \phi\gamma(h-1)$$

因此：
$$\gamma(h) = \phi^{h-1}\gamma(1)$$
$$\rho(h) = \phi^{h-1}\rho(1)$$

**注意：** $\rho(1)$本身不仅仅是$\phi$——它同时依赖$\phi$和$\theta$。

<div class="pitfall">
<strong>常见陷阱：</strong> 假设ARMA(1,1)滞后1处的ACF等于$\phi$。MA成分修改$\rho(1)$；只有后续滞后遵循纯AR(1)衰减。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 推导ARMA(1,1)同时平稳和可逆的条件。</summary>

<div class="answer">
<strong>答案：</strong> 平稳性要求$|\phi| < 1$；可逆性要求$|\theta| < 1$。

<strong>推导：</strong>

**平稳性：**
AR多项式：$\Phi(z) = 1 - \phi z$
根：$z = 1/\phi$
在单位圆外：$|1/\phi| > 1 \Rightarrow |\phi| < 1$

**可逆性：**
MA多项式：$\Theta(z) = 1 + \theta z$
根：$z = -1/\theta$
在单位圆外：$|{-1/\theta}| > 1 \Rightarrow |\theta| < 1$

**组合：** 过程平稳且可逆当且仅当$|\phi| < 1$且$|\theta| < 1$。

<div class="pitfall">
<strong>常见陷阱：</strong> 只检查平稳性。平稳但不可逆的ARMA有不适当的AR(∞)表示，导致预测和诊断问题。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你拟合了几个模型得到：AR(2) AIC=520, MA(2) AIC=525, ARMA(1,1) AIC=515, ARMA(2,1) AIC=514。你选择哪个模型，为什么？</summary>

<div class="answer">
<strong>答案：</strong> 选择ARMA(1,1)，尽管ARMA(2,1)有略低的AIC。

<strong>理由：</strong>
1. AIC差异1是可忽略的（在噪声范围内）
2. 简洁性原则：性能相似时优先选择更简单的模型
3. ARMA(1,1)更稳定、更可解释
4. 额外的AR参数不太可能改善预测

**决策框架：**
- AIC差异 < 2：模型基本等价
- AIC差异 2-7：有些证据支持低AIC模型
- AIC差异 > 10：强证据支持低AIC模型

还要考虑：
- BIC（对复杂性惩罚更大）
- 样本外预测准确度
- 所有候选的残差诊断

<div class="pitfall">
<strong>常见陷阱：</strong> 盲目选择最低AIC。小的AIC差异没有意义。始终考虑简洁性并用保留数据验证。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. 第4章。
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第4章。
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. 第2章。
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. 第3章。
