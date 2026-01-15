# 经典分解

<div class="interview-summary">
<strong>面试要点：</strong> 经典分解使用移动平均将序列分离为趋势、季节性和不规则组件。加法模型：$y = T + S + I$。乘法模型：$y = T \times S \times I$。简单但假设固定的季节模式且对异常值敏感。趋势通过中心化移动平均提取；季节性通过周期平均提取。
</div>

## 核心定义

**加法模型：**
$$y_t = T_t + S_t + I_t$$

当季节变化与水平无关时使用。

**乘法模型：**
$$y_t = T_t \times S_t \times I_t$$

当季节变化随水平缩放时使用。

**组件：**
- $T_t$：趋势-周期（平滑的基础水平）
- $S_t$：季节性（每 $m$ 个周期重复的周期模式）
- $I_t$：不规则/残差（随机噪声）

**季节性指数：**
- 加法：$S_t$ 值在一个周期内总和为0
- 乘法：$S_t$ 值在一个周期内平均为1

## 数学原理与推导

### 通过移动平均提取趋势

**中心化移动平均（CMA）：**

对于奇数 $m$（如 $m=7$）：
$$T_t = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} y_{t+j}$$

对于偶数 $m$（如 $m=12$）：
$$T_t = \frac{1}{2m}\left(y_{t-m/2} + 2\sum_{j=-(m/2-1)}^{m/2-1} y_{t+j} + y_{t+m/2}\right)$$

这是 $2 \times m$-MA：先 $m$-MA，再2-MA来中心化。

### 季节性指数计算

**加法：**
1. 去趋势：$y_t - T_t$
2. 对每个季节的去趋势值取平均：$\bar{S}_s = \frac{1}{k}\sum_{j} (y_{s+jm} - T_{s+jm})$
3. 归一化：$S_s = \bar{S}_s - \frac{1}{m}\sum_{s=1}^{m}\bar{S}_s$

**乘法：**
1. 去趋势：$y_t / T_t$
2. 对每个季节的比率取平均：$\bar{S}_s = \frac{1}{k}\sum_{j} \frac{y_{s+jm}}{T_{s+jm}}$
3. 归一化：$S_s = \bar{S}_s \times \frac{m}{\sum_{s=1}^{m}\bar{S}_s}$

### 移动平均的性质

$m$点移动平均：
- 消除周期为 $m$ 的季节性（在完整周期上取平均）
- 平滑高频噪声
- 引入 $(m-1)/2$ 个周期的滞后
- 在两端各损失 $(m-1)/2$ 个观测值

**频率响应：** MA是低通滤波器，衰减频率 $\geq 1/m$。

## 算法/模型概述

**经典分解算法：**

```
输入：y[1:n]，季节周期m，类型（加法/乘法）
输出：趋势T，季节性S，不规则I

1. 趋势提取
   - 计算y的中心化移动平均
   - T[t] = CMA(y, m)，对于 t = m/2+1 到 n-m/2
   - 端点：使用外推或保留缺失

2. 去趋势
   - 加法：D[t] = y[t] - T[t]
   - 乘法：D[t] = y[t] / T[t]

3. 季节性指数
   - 按季节分组D[t]（1到m）
   - 对每组取平均：S_raw[s] = mean(D[s], D[s+m], D[s+2m],...)
   - 归一化：
     - 加法：S[s] = S_raw[s] - mean(S_raw)
     - 乘法：S[s] = S_raw[s] × m / sum(S_raw)

4. 季节性组件
   - S[t] = S[t mod m]（在序列中复制指数）

5. 不规则组件
   - 加法：I[t] = y[t] - T[t] - S[t]
   - 乘法：I[t] = y[t] / (T[t] × S[t])

返回 T, S, I
```

## 常见陷阱

1. **固定季节模式**：经典分解假设整个过程中相同的季节模式。不能适应演化的季节性。

2. **异常值敏感性**：一个异常值影响趋势（通过MA）和季节性指数。没有稳健拟合。

3. **端点损失**：在两端各损失 $m/2$ 个观测值。对短序列问题大。

4. **模型类型错误**：当数据是乘法时使用加法（或反之）会导致分解效果差。

5. **日历效应**：不处理交易日、复活节等。这些出现在不规则组件中。

6. **非整数周期**：需要整数 $m$。对于365.25天/年，需要替代方法。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# 生成乘法季节性数据
np.random.seed(42)
n = 48  # 4年月度数据
t = np.arange(n)
trend = 100 + 2 * t
seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
y = trend * seasonal_mult * (1 + 0.05 * np.random.randn(n))

# 经典分解（乘法）
result = seasonal_decompose(y, model='multiplicative', period=12)

print("季节性指数（应该重复）:")
print(np.round(result.seasonal[:12], 3))

print("\n趋势（第一个和最后一个可用值）:")
print(f"  第一个: {result.trend[~np.isnan(result.trend)][0]:.1f}")
print(f"  最后一个: {result.trend[~np.isnan(result.trend)][-1]:.1f}")

# 比较加法（对此数据是错误的模型）
result_add = seasonal_decompose(y, model='additive', period=12)
print("\n比较残差标准差:")
print(f"  乘法: {np.nanstd(result.resid):.3f}")
print(f"  加法: {np.nanstd(result_add.resid):.3f}")
# 乘法应该有更小的残差标准差
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念）：</strong> 什么时候应该使用加法分解vs乘法分解？</summary>

<div class="answer">
<strong>答案：</strong>

**加法**适用于：
- 季节波动在绝对值上大致恒定
- 低值和高值有相似的季节波动
- 示例：温度（无论基础温度如何都是±10°F）

**乘法**适用于：
- 季节波动与水平成比例
- 百分比变化恒定
- 示例：零售销售（无论总销售额如何，12月都高于平均值20%）

**决策测试：**
1. 绘制序列——季节波动是否随水平增长？
2. 计算：不同时期的 std(季节性) / mean(水平)
   - 如果比率恒定 → 乘法
   - 如果 std(季节性) 恒定 → 加法

<div class="pitfall">
<strong>常见陷阱：</strong> 默认使用加法。许多经济/商业序列是乘法的，因为增长是复合的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念）：</strong> 为什么移动平均消除周期为m的季节性？</summary>

<div class="answer">
<strong>答案：</strong> $m$点移动平均恰好包含一个完整的季节周期。由于季节性组件在一个周期内总和为零（加法）或平均为一（乘法），它们相互抵消。

**数学解释（加法）：**
$$\text{MA}_t = \frac{1}{m}\sum_{j=0}^{m-1}(T_{t+j} + S_{t+j} + I_{t+j})$$

如果趋势局部恒定且 $\sum_{j=0}^{m-1}S_{t+j} = 0$：
$$\text{MA}_t \approx T_t + \frac{1}{m}\sum_{j=0}^{m-1}I_{t+j}$$

季节性抵消；只剩下趋势和平滑后的噪声。

<div class="pitfall">
<strong>常见陷阱：</strong> 使用错误的MA阶数。如果真实周期是12但你使用MA(6)，季节性不会被完全去除。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学）：</strong> 对于月度数据的12点中心化移动平均，两端各损失多少观测值？</summary>

<div class="answer">
<strong>答案：</strong> 两端各损失6个观测值（共12个）。

**推导：**
对于偶数 $m=12$，时间 $t$ 处的 2×12-MA 公式使用：
$$\frac{1}{24}(y_{t-6} + 2y_{t-5} + \cdots + 2y_{t+5} + y_{t+6})$$

这需要从 $t-6$ 到 $t+6$ 的观测值。

在开始处：只能对 $t \geq 7$ 计算（需要 $y_1, \ldots, y_{13}$）
在结尾处：只能对 $t \leq n-6$ 计算（需要数据到 $y_{n}$）

所以位置1-6和(n-5)-n缺失 → 共12个缺失值。

<div class="pitfall">
<strong>常见陷阱：</strong> 对于短序列（比如2年=24个点），损失12个点意味着一半的数据没有趋势估计。考虑使用STL或参数化趋势代替。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学）：</strong> 为什么季节性指数必须归一化，它们满足什么约束？</summary>

<div class="answer">
<strong>答案：</strong>

**为什么归一化：**
没有归一化，原始季节性指数的平均值可能不是零（加法）或一（乘法），导致趋势或不规则组件的系统偏差。

**约束：**

加法：$\sum_{s=1}^{m} S_s = 0$
- 确保季节性不移动整体水平
- 正季节被负季节抵消

乘法：$\sum_{s=1}^{m} S_s = m$（等价地，平均值=1）
- 确保季节因子不膨胀/缩小整体水平
- 大于1的因子被小于1的因子抵消

**归一化公式：**
- 加法：$S_s^{new} = S_s^{raw} - \bar{S}^{raw}$
- 乘法：$S_s^{new} = S_s^{raw} \times m / \sum S^{raw}$

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记归一化导致分解中 $T + S + I \neq y$，因为季节性有偏差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践）：</strong> 你分解月度销售数据，发现不规则组件在滞后1处有强自相关。这说明什么，如何解决？</summary>

<div class="answer">
<strong>答案：</strong> 不规则组件的滞后1自相关表明：
1. 模型没有捕获所有系统性模式
2. 存在超出趋势和季节性的短期动态
3. 可能：不在季节模式中的月度动量

**如何解决：**

1. **建模不规则组件：** 对不规则组件拟合ARIMA
   - 如果是AR(1)，纳入预测
   - STL+ARIMA或类似流程

2. **使用更好的分解：** STL可以适应经典方法遗漏的变化模式

3. **考虑直接使用SARIMA：** 在一个模型中处理趋势、季节性和自相关

4. **检查日历效应：** 交易日、节假日可能产生自相关

5. **使用ETS：** 状态空间模型可以捕获自相关误差

<div class="pitfall">
<strong>常见陷阱：</strong> 在预测中忽略不规则组件的自相关。这低估预测不确定性并可能使预测产生偏差。
</div>
</div>
</details>

## 参考文献

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 3.
2. Makridakis, S., Wheelwright, S. C., & Hyndman, R. J. (1998). *Forecasting: Methods and Applications*. Wiley. Chapter 4.
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 1.
4. Census Bureau. (2017). X-13ARIMA-SEATS Reference Manual. U.S. Census Bureau.
