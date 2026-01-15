# 自相关与偏自相关

<div class="interview-summary">
<strong>面试摘要：</strong> ACF测量序列与其滞后值之间的相关性。PACF测量去除中间滞后影响后滞后k处的相关性。ACF/PACF模式用于识别AR/MA阶数：AR(p)的PACF在滞后p处截尾；MA(q)的ACF在滞后q处截尾。在白噪声假设下，样本ACF/PACF的近似标准误为$1/\sqrt{n}$。
</div>

## 核心定义

**自协方差函数（ACVF）**：对于均值为$\mu$的平稳过程：
$$\gamma(h) = \text{Cov}(X_t, X_{t+h}) = E[(X_t - \mu)(X_{t+h} - \mu)]$$

**自相关函数（ACF）**：标准化的自协方差：
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \text{Corr}(X_t, X_{t+h})$$

**偏自相关函数（PACF）**：去除对$X_{t+1}, \ldots, X_{t+h-1}$的线性依赖后，$X_t$与$X_{t+h}$之间的相关性：
$$\phi_{hh} = \text{Corr}(X_t - \hat{X}_t, X_{t+h} - \hat{X}_{t+h})$$

其中$\hat{X}_t$和$\hat{X}_{t+h}$是基于中间值的最佳线性预测。

**样本ACF**：从数据估计：
$$\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)} = \frac{\sum_{t=1}^{n-h}(X_t - \bar{X})(X_{t+h} - \bar{X})}{\sum_{t=1}^{n}(X_t - \bar{X})^2}$$

## 数学与推导

### ACF的性质

对于任何平稳过程：

1. $\rho(0) = 1$
2. $\rho(h) = \rho(-h)$（对称性）
3. $|\rho(h)| \leq 1$ 对所有$h$成立
4. $\rho(h)$是半正定的：对于任意$a_1, \ldots, a_n$：
   $$\sum_{i=1}^{n}\sum_{j=1}^{n} a_i a_j \rho(i-j) \geq 0$$

### 通过Yule-Walker方程求PACF

滞后$k$处的PACF是AR(k)回归中的最后一个系数$\phi_{kk}$：
$$X_t = \phi_{k1}X_{t-1} + \phi_{k2}X_{t-2} + \cdots + \phi_{kk}X_{t-k} + \epsilon_t$$

矩阵形式的Yule-Walker方程：
$$\begin{pmatrix} 1 & \rho(1) & \cdots & \rho(k-1) \\ \rho(1) & 1 & \cdots & \rho(k-2) \\ \vdots & & \ddots & \vdots \\ \rho(k-1) & \cdots & \rho(1) & 1 \end{pmatrix} \begin{pmatrix} \phi_{k1} \\ \phi_{k2} \\ \vdots \\ \phi_{kk} \end{pmatrix} = \begin{pmatrix} \rho(1) \\ \rho(2) \\ \vdots \\ \rho(k) \end{pmatrix}$$

### AR(1)的ACF：$X_t = \phi X_{t-1} + \epsilon_t$

$$\rho(h) = \phi^{|h|}$$

当$|\phi| < 1$时，ACF呈指数（几何）衰减。

### MA(1)的ACF：$X_t = \epsilon_t + \theta\epsilon_{t-1}$

$$\rho(1) = \frac{\theta}{1+\theta^2}, \quad \rho(h) = 0 \text{ 当 } h > 1$$

ACF在滞后1之后截尾。

### MA(q)的ACF

$$\rho(h) = 0 \text{ 当 } h > q$$

### AR(p)的PACF

$$\phi_{hh} = 0 \text{ 当 } h > p$$

### 样本ACF的方差

在真实过程为白噪声的零假设下：
$$\text{Var}(\hat{\rho}(h)) \approx \frac{1}{n}$$

因此近似95%置信带为$\pm 1.96/\sqrt{n}$。

**Bartlett公式**（对于MA(q)过程）：
$$\text{Var}(\hat{\rho}(h)) \approx \frac{1}{n}\left(1 + 2\sum_{k=1}^{q}\rho(k)^2\right) \text{ 当 } h > q$$

## 算法/模型概述

**使用ACF/PACF进行模型识别：**

| 模式 | ACF | PACF | 模型 |
|---------|-----|------|-------|
| AR(p) | 指数/正弦衰减 | 在滞后p处截尾 | AR(p) |
| MA(q) | 在滞后q处截尾 | 指数/正弦衰减 | MA(q) |
| ARMA(p,q) | 拖尾 | 拖尾 | ARMA(p,q) |
| 白噪声 | 全部接近零 | 全部接近零 | 无需建模 |
| 非平稳 | 衰减非常慢 | 滞后1处有大尖峰 | 先差分 |

**解释步骤：**

```
1. 绘制序列 - 检查平稳性
2. 如果非平稳，差分直到平稳
3. 计算样本ACF和PACF
4. 检查显著的尖峰（超出±1.96/√n带）
5. 识别模式：
   - ACF截尾，PACF衰减 → MA(q)，其中q=截尾滞后
   - PACF截尾，ACF衰减 → AR(p)，其中p=截尾滞后
   - 两者都衰减 → ARMA（使用信息准则）
6. 拟合候选模型
7. 检查残差ACF/PACF（应为白噪声）
```

## 常见陷阱

1. **忽略置信带**：不是所有尖峰都是显著的。使用$\pm 1.96/\sqrt{n}$带，预期约5%的尖峰会偶然超出。

2. **混淆"截尾"和"拖尾"**：截尾意味着在滞后q之后突然降至零。拖尾意味着逐渐衰减。这种区别决定了AR还是MA。

3. **对非平稳数据应用ACF/PACF**：对非平稳序列的结果没有意义。始终先检查平稳性。

4. **过度解释高滞后相关**：对于小样本，高滞后估计有很高的方差。关注早期滞后。

5. **忘记季节性滞后**：在季节性数据中，检查季节周期的滞后（例如，对于具有年度季节性的月度数据，检查滞后12）。

6. **忽视理论ACF/PACF**：验证模型时，将样本函数与理论函数比较，而不仅仅是残差。

## 简单示例

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

# 模拟AR(2)过程
np.random.seed(42)
ar_params = np.array([1, -0.75, 0.25])  # 1 - 0.75L + 0.25L^2
ma_params = np.array([1])
ar2_process = ArmaProcess(ar_params, ma_params)
ar2_data = ar2_process.generate_sample(nsample=300)

# 绘制ACF和PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ar2_data, ax=axes[0], lags=20, title='AR(2)的ACF')
plot_pacf(ar2_data, ax=axes[1], lags=20, title='AR(2)的PACF')
plt.tight_layout()
plt.show()

# AR(2)：PACF应在滞后2后截尾，ACF应衰减
# 检查显著PACF值
from statsmodels.tsa.stattools import pacf
pacf_values = pacf(ar2_data, nlags=5)
print("PACF值:", pacf_values)
# 预期：滞后1,2处显著；之后接近零
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> ACF和PACF的根本区别是什么？为什么我们需要两者？</summary>

<div class="answer">
<strong>答案：</strong> ACF测量$X_t$和$X_{t+h}$之间的总相关性，包括通过中间滞后的间接效应。PACF测量去除中间效应后的直接相关性。

<strong>解释：</strong> 考虑$\phi = 0.8$的AR(1)。ACF显示$\rho(2) = 0.64$，因为$X_t$和$X_{t+2}$通过$X_{t+1}$相关。但滞后2处的PACF接近零，因为一旦我们考虑了$X_{t+1}$，就没有额外的直接关系。

我们需要两者是因为：
- ACF识别MA阶数（在滞后q处截尾）
- PACF识别AR阶数（在滞后p处截尾）

<div class="pitfall">
<strong>常见陷阱：</strong> 仅使用ACF进行识别。没有PACF，你无法区分AR和MA模式——两者都可能显示衰减的ACF，但只有AR显示PACF截尾。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 如何解释ACF显示非常慢的衰减，值在滞后20之后仍然显著？</summary>

<div class="answer">
<strong>答案：</strong> 这强烈表明非平稳性。平稳过程的ACF应该相对快速地衰减到零。非常慢的衰减表明存在单位根或近单位根。

<strong>建议操作：</strong>
1. 用ADF/KPSS正式检验
2. 对序列差分
3. 差分后重新计算ACF
4. 差分后的序列应显示更快的衰减

**关键洞察：** 对于随机游走，在有限样本中$\rho(h) \approx 1$对所有$h$成立，因为连续的值高度相关。

<div class="pitfall">
<strong>常见陷阱：</strong> 试图对非平稳数据拟合ARMA模型。结果参数会产生误导，预测也会很差。始终先确保平稳性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 推导MA(1)的ACF：$X_t = \epsilon_t + \theta\epsilon_{t-1}$。</summary>

<div class="answer">
<strong>答案：</strong> $\rho(0) = 1$，$\rho(1) = \frac{\theta}{1+\theta^2}$，$\rho(h) = 0$ 当 $h \geq 2$。

<strong>推导：</strong>

**方差（滞后0）：**
$$\gamma(0) = \text{Var}(X_t) = \text{Var}(\epsilon_t + \theta\epsilon_{t-1}) = \sigma^2 + \theta^2\sigma^2 = (1+\theta^2)\sigma^2$$

**滞后1的自协方差：**
$$\gamma(1) = \text{Cov}(X_t, X_{t+1}) = \text{Cov}(\epsilon_t + \theta\epsilon_{t-1}, \epsilon_{t+1} + \theta\epsilon_t)$$
$$= \text{Cov}(\epsilon_t, \theta\epsilon_t) = \theta\sigma^2$$

**滞后$h \geq 2$的自协方差：**
$$\gamma(h) = \text{Cov}(\epsilon_t + \theta\epsilon_{t-1}, \epsilon_{t+h} + \theta\epsilon_{t+h-1}) = 0$$

（当$h \geq 2$时没有重叠的$\epsilon$项）

**ACF：**
$$\rho(1) = \frac{\gamma(1)}{\gamma(0)} = \frac{\theta\sigma^2}{(1+\theta^2)\sigma^2} = \frac{\theta}{1+\theta^2}$$

<div class="pitfall">
<strong>常见陷阱：</strong> 注意MA(1)的$|\rho(1)| \leq 0.5$。最大值在$\theta = \pm 1$时出现。如果你观察到$|\hat{\rho}(1)| > 0.5$，可能是AR而不是MA。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 对于AR(2)过程$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$，证明PACF对所有滞后$h > 2$为零。</summary>

<div class="answer">
<strong>答案：</strong> 对于AR(p)，PACF $\phi_{hh} = 0$ 当 $h > p$，因为一旦包含$X_{t-1}, \ldots, X_{t-p}$，添加更多滞后不提供额外的预测信息。

<strong>解释：</strong>

AR(2)模型表明$X_t$只依赖于$X_{t-1}$和$X_{t-2}$（加上噪声）。因此：

对于$h = 3$：我们在$X_{t-1}, X_{t-2}, X_{t-3}$上回归$X_t$。
- $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$
- $X_{t-3}$只通过$X_{t-2}$和$X_{t-1}$影响$X_t$
- 在条件于$X_{t-1}$和$X_{t-2}$之后，$X_{t-3}$不增加信息
- 因此$\phi_{33} = 0$

通过归纳，这对所有$h > 2$成立。

**关键方程：** 滞后$k$处的PACF是使用恰好$k$个滞后的最佳线性预测中的系数$\phi_{kk}$。对于AR(p)，当$k > p$时第$k$个系数变为零。

<div class="pitfall">
<strong>常见陷阱：</strong> 期望样本PACF精确为零。由于抽样变异性，$\hat{\phi}_{hh}$将非零，但对于$h > p$应落在置信带内。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你为$n=100$个观测值的时间序列计算样本ACF。你看到滞后1、2、7和15超出95%置信带。你的解释是什么？</summary>

<div class="answer">
<strong>答案：</strong> 使用95%置信带测试约15-20个滞后，预期约有1个滞后偶然显著。滞后1和2可能是真实信号。滞后7可能是真实的（如果相关的话检查周模式）。滞后15可能是虚假的。

<strong>解释过程：</strong>
1. 关注早期滞后（1、2、3）——最可能真实
2. 考虑领域知识（滞后7 = 周？滞后12 = 月？）
3. 孤立的高滞后尖峰通常是噪声
4. 显著滞后的模式比单个尖峰更重要
5. 滞后15在$n=100$时有很高的方差：$\text{SE} \approx 1/\sqrt{100} = 0.1$，且只有约85对数据点

**置信带：** $\pm 1.96/\sqrt{100} = \pm 0.196$

<div class="pitfall">
<strong>常见陷阱：</strong> 将每个显著滞后视为有意义的。有许多滞后时，会出现假阳性。使用序贯检验修正或关注有意义的模式，而非孤立的尖峰。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley. 第2章。
2. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. 第3章。
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第2-3章。
4. Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time-series. *JRSS B*, 8(1), 27-41.
