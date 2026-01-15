# 自回归（AR）模型

<div class="interview-summary">
<strong>面试摘要：</strong> AR(p)模型将当前值表示为p个过去值加噪声的线性组合。平稳性要求特征多项式的根在单位圆外（等价地，AR(1)要求$|\phi| < 1$）。ACF呈指数/正弦衰减；PACF在滞后p处截尾。通过Yule-Walker、OLS或MLE估计。
</div>

## 核心定义

**AR(p)模型**：
$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

其中$\epsilon_t \sim WN(0, \sigma^2)$（白噪声）。

**滞后算子形式**：
$$\Phi(L)X_t = c + \epsilon_t$$

其中$\Phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p$，且$LX_t = X_{t-1}$。

**特征多项式**：
$$\Phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p$$

**平稳性条件**：$\Phi(z) = 0$的所有根必须在单位圆外（|z| > 1）。

**平稳AR(p)的均值**：
$$\mu = E[X_t] = \frac{c}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}$$

## 数学与推导

### AR(1)模型：$X_t = c + \phi X_{t-1} + \epsilon_t$

**平稳性条件**：$|\phi| < 1$

**均值**：$\mu = \frac{c}{1-\phi}$

**方差**：
$$\gamma(0) = \text{Var}(X_t) = \frac{\sigma^2}{1-\phi^2}$$

**自协方差**：
$$\gamma(h) = \phi^{|h|} \gamma(0) = \frac{\phi^{|h|} \sigma^2}{1-\phi^2}$$

**ACF**：$\rho(h) = \phi^{|h|}$

**PACF**：$\phi_{11} = \phi$，$\phi_{hh} = 0$ 当 $h > 1$

### AR(2)模型：$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$

**平稳性条件**（必须全部满足）：
1. $\phi_1 + \phi_2 < 1$
2. $\phi_2 - \phi_1 < 1$
3. $|\phi_2| < 1$

**特征根**：$1 - \phi_1 z - \phi_2 z^2 = 0$的解
- 如果根是实数：ACF单调衰减
- 如果根是复数：ACF阻尼正弦衰减

**AR(2)的Yule-Walker方程**：
$$\rho(1) = \phi_1 + \phi_2\rho(1) \Rightarrow \rho(1) = \frac{\phi_1}{1-\phi_2}$$
$$\rho(2) = \phi_1\rho(1) + \phi_2$$

### 一般AR(p)的Yule-Walker方程

$$\gamma(h) = \phi_1\gamma(h-1) + \phi_2\gamma(h-2) + \cdots + \phi_p\gamma(h-p) \text{ 当 } h > 0$$

矩阵形式：
$$\begin{pmatrix} \gamma(0) & \gamma(1) & \cdots & \gamma(p-1) \\ \gamma(1) & \gamma(0) & \cdots & \gamma(p-2) \\ \vdots & & \ddots & \vdots \\ \gamma(p-1) & \cdots & \gamma(1) & \gamma(0) \end{pmatrix} \begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{pmatrix} = \begin{pmatrix} \gamma(1) \\ \gamma(2) \\ \vdots \\ \gamma(p) \end{pmatrix}$$

或用自相关表示：
$$\mathbf{R}\boldsymbol{\phi} = \boldsymbol{\rho}$$

### 无限MA表示

平稳AR(p)可以写成无限MA：
$$X_t = \mu + \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}$$

对于AR(1)：$\psi_j = \phi^j$

这表明AR过程有无限记忆，但权重呈指数衰减。

## 算法/模型概述

**估计方法：**

1. **Yule-Walker（矩方法）**：
   - 用$\hat{\gamma}(h)$替换$\gamma(h)$
   - 求解线性系统得到$\hat{\phi}$
   - 总是得到平稳估计
   - 对小样本可能不够有效

2. **普通最小二乘（OLS）**：
   - 在$X_{t-1}, \ldots, X_{t-p}$上回归$X_t$
   - 简单但会丢失前$p$个观测值
   - 可能给出非平稳估计

3. **最大似然（MLE）**：
   - 渐近最有效
   - 考虑初始条件
   - 需要分布假设（通常是高斯）
   - 使用数值优化

**阶数选择：**
```
1. 检查PACF - 显著尖峰表明AR阶数
2. 拟合AR(1), AR(2), ..., AR(p_max)
3. 比较AIC/BIC值
4. 选择信息准则最低的模型
5. 验证残差是白噪声
```

## 常见陷阱

1. **忽略平稳性检查**：始终验证估计参数满足平稳性条件。非平稳AR导致爆炸性预测。

2. **使用过多滞后过拟合**：AIC可能偏好较大的模型。BIC对复杂性惩罚更大，通常给出更好的预测。

3. **假设因果关系**：AR模型捕获相关性，而非因果关系。$X_{t-1}$预测$X_t$不意味着过去导致未来。

4. **忽视季节性**：标准AR不能捕获滞后$s$处的季节性模式。考虑SARIMA或显式包含$X_{t-s}$。

5. **使用OLS而不修正**：标准OLS标准误对有自相关的时间序列无效。使用HAC标准误或适当的基于似然的推断。

6. **混淆AR(1)系数符号**：正的$\phi$给出正自相关（动量）。负的$\phi$给出交替符号（均值回归）。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf

# 生成AR(2)数据
np.random.seed(42)
n = 300
phi1, phi2 = 0.6, -0.3
X = np.zeros(n)
eps = np.random.randn(n)

for t in range(2, n):
    X[t] = phi1 * X[t-1] + phi2 * X[t-2] + eps[t]

# 检查PACF（应在滞后2后截尾）
pacf_values = pacf(X, nlags=10)
print("PACF:", np.round(pacf_values, 3))

# 使用AIC选择阶数拟合AR模型
from statsmodels.tsa.ar_model import ar_select_order
sel = ar_select_order(X, maxlag=10, ic='aic')
print(f"选择的阶数: {sel.ar_lags}")

# 拟合AR(2)并检查估计
model = AutoReg(X, lags=2).fit()
print(f"真实值: phi1={phi1}, phi2={phi2}")
print(f"估计值: phi1={model.params[1]:.3f}, phi2={model.params[2]:.3f}")

# 预测
forecast = model.forecast(steps=5)
print("5步预测:", forecast)
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 直观解释为什么AR(1)的平稳性条件是$|\phi| < 1$。当$\phi = 1$或$\phi > 1$时会发生什么？</summary>

<div class="answer">
<strong>答案：</strong> 当$|\phi| < 1$时，冲击随时间衰减，保持方差有界。当$\phi = 1$时，我们有随机游走，冲击永久持续（单位根）。当$|\phi| > 1$时，过程呈指数爆炸。

<strong>解释：</strong>
AR(1)可以写成：
$$X_t = \phi^t X_0 + \sum_{j=0}^{t-1} \phi^j \epsilon_{t-j}$$

- 如果$|\phi| < 1$：$\phi^t \to 0$且MA表示收敛（有界方差）
- 如果$\phi = 1$：$X_t = X_0 + \sum \epsilon_j$（随机游走，方差$\to \infty$）
- 如果$|\phi| > 1$：$\phi^t \to \infty$（爆炸）

<div class="pitfall">
<strong>常见陷阱：</strong> 认为$\phi = 0.99$"足够接近"平稳。虽然技术上平稳，但近单位根过程在有限样本中表现得像随机游走。预测会迅速退化。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 为什么AR(p)过程的PACF在滞后p后截尾，而ACF逐渐衰减？</summary>

<div class="answer">
<strong>答案：</strong> PACF测量控制中间滞后后的直接相关性。AR(p)按定义只对$p$个过去值有直接依赖，所以PACF在滞后$p$之后为零。ACF包含通过中间值的间接效应，导致逐渐衰减。

<strong>解释：</strong>
对于AR(2)：$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$
- 直接效应：只来自$X_{t-1}$和$X_{t-2}$
- 滞后3处的PACF：控制$X_{t-1}, X_{t-2}$后，$X_{t-3}$没有额外预测能力
- 但滞后3处的ACF：$X_t$通过链条$X_{t-1} \to X_{t-2} \to X_{t-3}$与$X_{t-3}$相关

<div class="pitfall">
<strong>常见陷阱：</strong> 期望样本中滞后$p$之后PACF完全为零。由于估计误差，你会看到小的非零值。使用置信带判断显著性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 推导平稳AR(1)过程的方差：$\text{Var}(X_t) = \frac{\sigma^2}{1-\phi^2}$。</summary>

<div class="answer">
<strong>答案：</strong> 从$X_t = \phi X_{t-1} + \epsilon_t$开始，对两边取方差并利用平稳性。

<strong>推导：</strong>
$$\text{Var}(X_t) = \text{Var}(\phi X_{t-1} + \epsilon_t)$$
$$= \phi^2 \text{Var}(X_{t-1}) + \text{Var}(\epsilon_t) + 2\phi\text{Cov}(X_{t-1}, \epsilon_t)$$

由于$\epsilon_t$与$X_{t-1}$独立：
$$\gamma(0) = \phi^2 \gamma(0) + \sigma^2$$

根据平稳性，$\text{Var}(X_t) = \text{Var}(X_{t-1}) = \gamma(0)$：
$$\gamma(0) - \phi^2\gamma(0) = \sigma^2$$
$$\gamma(0)(1 - \phi^2) = \sigma^2$$
$$\gamma(0) = \frac{\sigma^2}{1-\phi^2}$$

**注意：** 要求$|\phi| < 1$才能得到正方差。

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记方差随$|\phi| \to 1$增加。近单位根过程有大方差，使它们看起来比相同$\sigma^2$的白噪声更不稳定。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 对于AR(2)，证明$(\phi_1, \phi_2)$平面上的平稳区域是一个三角形，顶点在$(2, -1)$、$(-2, -1)$和$(0, 1)$。</summary>

<div class="answer">
<strong>答案：</strong> 三个平稳性条件$\phi_1 + \phi_2 < 1$、$\phi_2 - \phi_1 < 1$和$|\phi_2| < 1$定义了一个三角形区域。

<strong>推导：</strong>
特征方程$1 - \phi_1 z - \phi_2 z^2 = 0$的根必须在单位圆外。

设$z = 1$：$1 - \phi_1 - \phi_2 > 0 \Rightarrow \phi_1 + \phi_2 < 1$
设$z = -1$：$1 + \phi_1 - \phi_2 > 0 \Rightarrow \phi_2 - \phi_1 < 1$

对于复根，判别式分析给出：$|\phi_2| < 1$

边界线：
- $\phi_1 + \phi_2 = 1$（过$(2, -1)$和$(0, 1)$）
- $\phi_2 - \phi_1 = 1$（过$(-2, -1)$和$(0, 1)$）
- $\phi_2 = -1$（连接$(2, -1)$和$(-2, -1)$）

<div class="pitfall">
<strong>常见陷阱：</strong> 只检查一个条件。三个条件必须同时满足。模型可能满足两个条件但不满足第三个，仍然是非平稳的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你拟合了AR(3)模型，得到估计$\hat{\phi}_1 = 0.5$、$\hat{\phi}_2 = 0.3$、$\hat{\phi}_3 = 0.25$。残差ACF在滞后1处显示显著尖峰。可能出了什么问题？</summary>

<div class="answer">
<strong>答案：</strong> 滞后1处的显著残差自相关表明模型设定错误。可能的原因：
1. 需要MA成分（ARMA而非纯AR）
2. 数据中的结构性断点
3. 影响估计的异常值
4. 非平稳性未完全处理

<strong>诊断步骤：</strong>
1. 对残差进行Ljung-Box检验
2. 尝试拟合ARMA(3,1)或ARMA(3,2)
3. 绘制残差随时间的图以检查模式
4. 检查异常值或水平移动
5. 验证原始序列是平稳的

**关键洞察：** 纯AR残差应该是白噪声。显著的残差自相关意味着模型没有捕获所有时间依赖性。

<div class="pitfall">
<strong>常见陷阱：</strong> 添加更多AR滞后来修复残差相关性。有时MA项更简洁。比较AR(4)、AR(5)和ARMA(3,1)的AIC。
</div>
</div>
</details>

## 参考文献

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第3章。
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. 第3章。
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第3章。
4. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. 第2章。
