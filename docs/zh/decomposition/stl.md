# STL分解

<div class="interview-summary">
<strong>面试要点：</strong> STL（使用Loess的季节性和趋势分解）使用局部回归将序列稳健地分离为趋势、季节性和残差组件。与经典分解不同，STL可以处理任意季节周期并且对异常值稳健。关键参数：季节窗口（≥7的奇数）和趋势窗口。残差应该是平稳的以便于预测。
</div>

## 核心定义

**加法分解：**
$$y_t = T_t + S_t + R_t$$

- $T_t$：趋势组件（平滑的长期变动）
- $S_t$：季节性组件（周期性模式，每个周期总和约为0）
- $R_t$：残差/剩余（其他所有内容）

**Loess（局部估计散点图平滑）：**
使用加权最小二乘的局部多项式回归。权重随与目标点距离的增加而减少。

**关键STL参数：**
- `seasonal`：季节周期（如月度数据为12）
- `seasonal_deg`：季节多项式的阶数（0或1）
- `trend`：趋势平滑窗口（奇数整数，默认取决于季节）
- `robust`：是否使用稳健拟合（降低异常值权重）

## 数学原理与推导

### Loess平滑器

对于点 $x_0$，拟合加权多项式：
$$\min_{\beta} \sum_{i=1}^{n} w_i(x_0)(y_i - \beta_0 - \beta_1(x_i - x_0))^2$$

权重使用三次方函数：
$$w(u) = \begin{cases} (1-|u|^3)^3 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}$$

距离按带宽 $h$ 缩放：$u_i = |x_i - x_0|/h$

### STL算法（简化）

**外循环**（用于稳健性）：
1. 初始化：$R_t^{(0)} = 0$，$T_t^{(0)} = $ $y_t$ 的loess平滑

**内循环**：
2. **去趋势**：$y_t - T_t^{(k-1)}$
3. **周期子序列平滑**：对于每个季节 $s=1,\ldots,m$，使用loess平滑位置 $s, s+m, s+2m, \ldots$ 处的值
4. **低通滤波器**：从季节性中去除低频
5. **去季节化**：$y_t - S_t^{(k)}$
6. **趋势提取**：去季节化序列的loess平滑

重复内循环直到收敛；外循环更新稳健性权重。

### 稳健性权重

每次外迭代后，计算残差并分配权重：
$$\rho_t = |R_t|$$
$$h = 6 \cdot \text{median}(|\rho_t|)$$
$$w_t = B(\rho_t/h)$$

其中 $B$ 是双权函数：对于 $|u| < 1$，$B(u) = (1-u^2)^2$，否则为0。

异常值在后续迭代中被降权。

## 算法/模型概述

**STL分解步骤：**

```
输入：y[1:n]，季节周期m，参数
输出：趋势T，季节性S，残差R

1. 初始化趋势 T = loess(y) 或移动平均
2. 对于 k = 1 到 n_outer：

   对于 j = 1 到 n_inner：
      a. 去趋势：D = y - T
      b. 对于位置 i 在 1...m 中：
         - 提取子序列：i, i+m, i+2m,... 处的值
         - 用loess平滑子序列
         - 存储平滑后的季节值
      c. 低通滤波季节性（去除趋势泄漏）
      d. 从原始季节性中减去滤波后的季节性 → S
      e. 去季节化：y - S
      f. 平滑去季节化 → T

   根据 R = y - T - S 更新稳健性权重

3. 返回 T, S, R = y - T - S
```

## 常见陷阱

1. **季节周期错误**：STL需要正确的m。如果m错误，季节性将无法正确捕获。

2. **趋势过度平滑**：趋势窗口过大会移除真实变化。平滑不足会捕获噪声。

3. **季节性泄漏到趋势**：如果季节窗口太小，趋势会吸收一些季节性。

4. **不使用稳健模式**：异常值会扭曲趋势和季节性。始终首先尝试`robust=True`。

5. **假设乘法直接有效**：STL是加法的。对于乘法，先取对数变换，然后分解，再取指数。

6. **忽略残差**：残差应该看起来像噪声。强模式表示模型不足。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# 生成带趋势、季节性和异常值的数据
np.random.seed(42)
n = 120
t = np.arange(n)
trend = 50 + 0.3 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(n) * 3
y = trend + seasonal + noise

# 添加异常值
y[50] += 40
y[80] -= 35

# STL分解（稳健）
stl = STL(y, period=12, robust=True)
result = stl.fit()

print("组件统计:")
print(f"趋势范围: [{result.trend.min():.1f}, {result.trend.max():.1f}]")
print(f"季节性范围: [{result.seasonal.min():.1f}, {result.seasonal.max():.1f}]")
print(f"残差标准差: {result.resid.std():.2f}")

# 检查异常值是否在残差中（应该是）
print(f"\n异常值位置的残差:")
print(f"  t=50: {result.resid[50]:.1f}")
print(f"  t=80: {result.resid[80]:.1f}")

# 绘图
fig = result.plot()
plt.tight_layout()
plt.show()
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念）：</strong> 为什么在许多应用中STL优于经典分解？</summary>

<div class="answer">
<strong>答案：</strong> STL的优势：

1. **灵活性**：适用于任何季节周期，不仅仅是4或12
2. **稳健性**：异常值不会扭曲估计（使用robust=True）
3. **可控性**：通过窗口参数调整平滑度
4. **演化季节性**：可以捕获缓慢变化的季节模式
5. **无端点问题**：Loess比移动平均更好地处理边界

经典分解：
- 假设固定的季节模式
- 对异常值敏感
- 移动平均在两端丢失观测值
- 仅限于标准频率

<div class="pitfall">
<strong>常见陷阱：</strong> 默认使用经典分解。STL几乎总是更好，特别是有异常值或演化模式时。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念）：</strong> STL中的稳健性机制是如何工作的？</summary>

<div class="answer">
<strong>答案：</strong> 迭代重新加权：

1. 第一轮：正常拟合STL，计算残差
2. 识别异常值：相对于中位数较大的|残差|
3. 分配权重：异常值权重 → 0，正常点 → 1
4. 用加权观测值重新拟合STL
5. 重复直到收敛

**权重函数（双权）：**
$$w = (1 - (r/h)^2)^2$$

其中 $r$ = |残差|，$h$ = 6 × median(|残差|)

异常值（大 $r$）获得接近零的权重，不影响拟合。

<div class="pitfall">
<strong>常见陷阱：</strong> 不使用稳健模式然后奇怪为什么一个异常值扭曲了整个季节模式。始终从robust=True开始。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学）：</strong> 在loess平滑中，为什么使用三次方权重函数？</summary>

<div class="answer">
<strong>答案：</strong> 三次方函数 $w(u) = (1-|u|^3)^3$：

1. **平滑**：连续可微，给出平滑的拟合曲线
2. **紧支撑**：在带宽之外为零，所以远点不影响拟合
3. **降权**：平滑地随距离减少影响
4. **计算友好**：简单的多项式形式

**性质：**
- $w(0) = 1$（目标点完全权重）
- $w(u) \to 0$ 随着 $|u| \to 1$ 平滑趋近
- $w(u) = 0$ 对于 $|u| \geq 1$

替代方案：高斯权重有无限支撑（所有点都有贡献），局部性较差。

<div class="pitfall">
<strong>常见陷阱：</strong> 带宽选择太小 → 拟合锯齿状；太大 → 过度平滑。STL使用数据驱动的默认值但调参可能有帮助。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学）：</strong> 为什么STL对季节性组件使用低通滤波器？</summary>

<div class="answer">
<strong>答案：</strong> 低通滤波器去除在周期子序列平滑期间泄漏到季节性中的趋势。

**问题：** 当平滑每个季节的子序列（如所有一月）时，如果存在趋势，子序列平均值会漂移。这种漂移作为低频内容出现在季节性中。

**解决方案：** 对跨完整周期的季节性应用移动平均：
$$L_t = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} S^*_{t+j}$$

然后减去：$S_t = S^*_t - L_t$

这确保季节性在每个周期内平均为零。

<div class="pitfall">
<strong>常见陷阱：</strong> 没有低通滤波器，季节性组件捕获一些趋势，留下带有趋势模式的残差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践）：</strong> STL分解后，你的残差显示清晰的AR(1)模式。这意味着什么，应该怎么做？</summary>

<div class="answer">
<strong>答案：</strong> 残差中的AR(1)模式意味着：
- STL捕获了趋势和季节性
- 但短期自相关仍然存在
- 这是常见且预期的

**怎么做：**
1. **为预测建模残差**：用AR(1)或ARIMA建模残差
   - 预测趋势（外推或漂移）
   - 预测季节性（重复模式）
   - 用AR(1)预测残差
   - 组合：$\hat{y} = \hat{T} + \hat{S} + \hat{R}$

2. **STL + ARIMA流程：**
   ```python
   stl_result = STL(y, period=12).fit()
   remainder = stl_result.resid
   arima_model = ARIMA(remainder, order=(1,0,0)).fit()
   ```

3. **考虑直接使用ETS/SARIMA**：它们在一个模型中处理所有组件。

<div class="pitfall">
<strong>常见陷阱：</strong> 在预测中忽略残差自相关。这会低估短期不确定性并使预测产生偏差。
</div>
</div>
</details>

## 参考文献

1. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
2. Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: An approach to regression analysis by local fitting. *JASA*, 83(403), 596-610.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 3.
4. Dokumentov, A., & Hyndman, R. J. (2015). STR: A seasonal-trend decomposition procedure based on regression. *Monash Econometrics Working Papers*.
