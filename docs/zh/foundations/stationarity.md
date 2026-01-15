# 平稳性

<div class="interview-summary">
<strong>面试摘要：</strong> 平稳性是大多数经典时间序列模型的基础假设。平稳过程具有常数均值、常数方差，且自协方差仅依赖于滞后期而非时间本身。弱（协方差）平稳通常就足够了。使用ADF、KPSS或PP检验进行测试。非平稳序列通常可以通过差分变为平稳。
</div>

## 核心定义

**严格（强）平稳性**：过程 $\{X_t\}$ 是严格平稳的，如果 $(X_{t_1}, X_{t_2}, \ldots, X_{t_k})$ 的联合分布与 $(X_{t_1+h}, X_{t_2+h}, \ldots, X_{t_k+h})$ 相同，对于所有 $k$、所有时间点 $t_1, \ldots, t_k$ 和所有位移 $h$。

**弱（协方差/二阶）平稳性**：过程是弱平稳的，如果：

1. $E[X_t] = \mu$（常数均值，有限）
2. $\text{Var}(X_t) = \sigma^2 < \infty$（常数方差，有限）
3. $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$（自协方差仅依赖于滞后 $h$）

**遍历性**（高层次）：遍历过程允许时间平均收敛到总体平均。这为从单个实现估计总体参数提供了理论依据。实践中遇到的大多数平稳过程都是遍历的。

**趋势平稳 vs. 差分平稳**：

- **趋势平稳**：$X_t = \mu_t + Y_t$，其中 $Y_t$ 是平稳的；通过回归去除趋势
- **差分平稳**：$\Delta X_t = X_t - X_{t-1}$ 是平稳的；通过差分去除单位根

## 数学与推导

### 自协方差函数

对于弱平稳过程：

$$\gamma(h) = \text{Cov}(X_t, X_{t+h}) = E[(X_t - \mu)(X_{t+h} - \mu)]$$

性质：
- $\gamma(0) = \text{Var}(X_t) = \sigma^2$
- $\gamma(h) = \gamma(-h)$（对称性）
- $|\gamma(h)| \leq \gamma(0)$（柯西-施瓦茨不等式）

### 自相关函数（ACF）

$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\text{Cov}(X_t, X_{t+h})}{\text{Var}(X_t)}$$

性质：
- $\rho(0) = 1$
- $|\rho(h)| \leq 1$
- $\rho(h) = \rho(-h)$

### 单位根与积分

如果 $(1-L)X_t$ 是平稳的，则过程存在单位根，其中 $L$ 是滞后算子（$LX_t = X_{t-1}$）。

**随机游走**（单位根示例）：
$$X_t = X_{t-1} + \epsilon_t$$

这是非平稳的：$\text{Var}(X_t) = t\sigma^2_\epsilon \to \infty$。

差分后：$\Delta X_t = \epsilon_t$ 是平稳的。

### 增广迪基-富勒（ADF）检验

检验单位根。模型：
$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

- $H_0$：$\gamma = 0$（存在单位根，非平稳）
- $H_1$：$\gamma < 0$（不存在单位根，平稳）

**KPSS检验**（互补检验）：
- $H_0$：序列是平稳的
- $H_1$：序列存在单位根

同时使用ADF和KPSS以获得稳健结论。

## 算法/模型概述

**平稳性检验：**

```
1. 目视检查：绘制序列图，寻找趋势/方差变化
2. ACF图：平稳序列的ACF衰减到零
3. ADF检验：拒绝H0 → 平稳
4. KPSS检验：无法拒绝H0 → 平稳
5. 如果非平稳：
   - 尝试差分（针对单位根）
   - 尝试去趋势（针对趋势平稳）
   - 检查是否需要季节差分
```

**使序列平稳：**

| 症状 | 解决方案 |
|---------|----------|
| 趋势（线性） | 一阶差分或去趋势 |
| 趋势（二次） | 二阶差分 |
| 季节性 | 季节差分 |
| 方差变化 | 对数变换，然后差分 |
| 趋势和季节性都有 | 组合变换 |

## 常见陷阱

1. **混淆严格和弱平稳性**：弱平稳通常足以进行ARIMA建模。严格平稳很少直接测试。

2. **过度差分**：对平稳序列进行差分会引入不必要的MA结构。检查ACF——如果已经在衰减，不要差分。

3. **忽略结构性断点**：具有结构性断点的序列可能看起来非平稳，但差分无法解决。考虑体制转换模型。

4. **误解ADF p值**：ADF检验的是单位根，而非平稳性。低p值拒绝单位根（表明平稳）。同时使用KPSS确认。

5. **忽视方差平稳性**：序列可以具有常数均值但方差变化（异方差性）。考虑GARCH模型或变换。

6. **季节性单位根**：标准ADF无法检测季节性单位根。使用HEGY检验或季节差分。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

# 生成随机游走（非平稳）
np.random.seed(42)
random_walk = np.cumsum(np.random.randn(200))

# 生成平稳AR(1)
ar1 = np.zeros(200)
for t in range(1, 200):
    ar1[t] = 0.7 * ar1[t-1] + np.random.randn()

# ADF检验
adf_rw = adfuller(random_walk)
print(f"随机游走 ADF p值: {adf_rw[1]:.4f}")  # 高 → 非平稳

adf_ar = adfuller(ar1)
print(f"AR(1) ADF p值: {adf_ar[1]:.4f}")  # 低 → 平稳

# 随机游走的一阶差分
diff_rw = np.diff(random_walk)
adf_diff = adfuller(diff_rw)
print(f"差分后RW ADF p值: {adf_diff[1]:.4f}")  # 低 → 平稳
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 严格平稳性和弱平稳性有什么区别？什么时候弱平稳性就足够了？</summary>

<div class="answer">
<strong>答案：</strong> 严格平稳性要求整个联合分布对时间位移不变。弱平稳性只要求常数均值、常数方差，以及自协方差仅依赖于滞后。

<strong>解释：</strong> 弱平稳性对于ARIMA类型模型是足够的，因为这些模型只使用一阶和二阶矩（均值和协方差）。参数估计或预测不需要完整的分布特性。

**关键点：** 如果一个过程是严格平稳的且具有有限的二阶矩，则它也是弱平稳的。反之则不一定成立。

<div class="pitfall">
<strong>常见陷阱：</strong> 假设弱平稳性意味着高斯性。弱平稳过程可以具有任何边际分布——它只约束前两阶矩。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 解释趋势平稳性和差分平稳性之间的区别。如何处理每种情况？</summary>

<div class="answer">
<strong>答案：</strong> 趋势平稳：序列具有确定性趋势；减去趋势得到平稳残差。差分平稳：序列具有随机趋势（单位根）；差分消除非平稳性。

<strong>解释：</strong>
- 趋势平稳：$X_t = \alpha + \beta t + Y_t$，其中 $Y_t$ 是平稳的。拟合趋势并减去。
- 差分平稳：$X_t = X_{t-1} + \epsilon_t$。一阶差分：$\Delta X_t = \epsilon_t$。

应用错误的变换是低效的——对趋势平稳序列进行差分会增加MA(1)结构；对差分平稳序列去趋势会留下自相关。

<div class="pitfall">
<strong>常见陷阱：</strong> 对所有情况都使用差分。始终绘制序列并考虑趋势是确定性的还是随机的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 证明对于随机游走 $X_t = X_{t-1} + \epsilon_t$，方差随时间线性增长。</summary>

<div class="answer">
<strong>答案：</strong> $\text{Var}(X_t) = t \cdot \sigma^2_\epsilon$

<strong>推导：</strong>
从 $X_0 = 0$ 开始：
$$X_t = \sum_{i=1}^{t} \epsilon_i$$

由于 $\epsilon_i$ 是独立的，方差为 $\sigma^2_\epsilon$：
$$\text{Var}(X_t) = \text{Var}\left(\sum_{i=1}^{t} \epsilon_i\right) = \sum_{i=1}^{t} \text{Var}(\epsilon_i) = t \cdot \sigma^2_\epsilon$$

这表明方差无界增长，违反了弱平稳性。

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记平稳过程的累积和通常是非平稳的。积分（求和）和微分对平稳性有相反的影响。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 推导AR(1)过程 $X_t = \phi X_{t-1} + \epsilon_t$（其中 $|\phi| < 1$）的自相关函数。</summary>

<div class="answer">
<strong>答案：</strong> $\rho(h) = \phi^{|h|}$

<strong>推导：</strong>
对于平稳性，将两边乘以 $X_{t-h}$ 并取期望：
$$E[X_t X_{t-h}] = \phi E[X_{t-1} X_{t-h}] + E[\epsilon_t X_{t-h}]$$

由于 $\epsilon_t$ 与过去的值不相关：
$$\gamma(h) = \phi \gamma(h-1)$$（对于 $h \geq 1$）

这是一个一阶递推关系，解为：
$$\gamma(h) = \phi^h \gamma(0)$$

因此：
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \phi^h$$

对于 $h < 0$，使用对称性：$\rho(h) = \phi^{|h|}$

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记平稳性条件 $|\phi| < 1$。如果 $|\phi| \geq 1$，过程会爆炸且没有有限方差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你运行ADF检验得到p值 = 0.08，KPSS检验得到p值 = 0.03。你得出什么结论？应该怎么做？</summary>

<div class="answer">
<strong>答案：</strong> 结果是矛盾的。ADF表明可能平稳（p = 0.08是边界值），但KPSS拒绝平稳性（p = 0.03）。这通常表明序列接近单位根或存在结构性断点。

<strong>建议操作：</strong>
1. 绘制序列和ACF进行目视检查
2. 尝试差分并重新检验
3. 检查结构性断点（Chow检验，CUSUM）
4. 考虑序列可能是分数积分的
5. 利用领域知识——是否预期非平稳性？

**关键公式：** 对于KPSS，低p值拒绝平稳性。对于ADF，低p值拒绝单位根（支持平稳性）。

<div class="pitfall">
<strong>常见陷阱：</strong> 依赖单一检验。ADF对接近单位根的备择假设功效较低。KPSS可能因序列相关而拒绝平稳性。始终使用多种方法和目视检查。
</div>
</div>
</details>

## 参考文献

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第3, 17章。
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第1章。
3. Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *JASA*, 74(366), 427-431.
4. Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity. *Journal of Econometrics*, 54(1-3), 159-178.
