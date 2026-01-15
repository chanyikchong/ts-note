# 简单指数平滑（SES）

<div class="interview-summary">
<strong>面试摘要：</strong> SES使用指数衰减权重的过去观测值加权平均进行预测。公式：$\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$。参数$\alpha \in (0,1)$控制响应性。等价于ARIMA(0,1,1)。最适合没有趋势或季节性的序列。所有预测步长的点预测是平的（常数）。
</div>

## 核心定义

**简单指数平滑**：
$$\hat{y}_{t+1|t} = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}$$

**替代形式：**

成分形式：
$$\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$$
$$\hat{y}_{t+h|t} = \ell_t$$

加权平均形式：
$$\hat{y}_{t+1|t} = \alpha \sum_{j=0}^{t-1}(1-\alpha)^j y_{t-j} + (1-\alpha)^t \ell_0$$

**参数：**
- $\alpha \in (0,1)$：平滑参数
- $\ell_0$：初始水平

## 数学与推导

### 指数权重

展开递推：
$$\hat{y}_{t+1|t} = \alpha y_t + \alpha(1-\alpha)y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots$$

权重：$\alpha, \alpha(1-\alpha), \alpha(1-\alpha)^2, \ldots$

这些权重和为1：$\alpha \sum_{j=0}^{\infty}(1-\alpha)^j = \alpha \cdot \frac{1}{1-(1-\alpha)} = 1$

### 与ARIMA(0,1,1)的联系

ARIMA(0,1,1)：$(1-L)y_t = (1+\theta L)\epsilon_t$

最优预测是：
$$\hat{y}_{t+1|t} = \hat{y}_{t|t-1} + \frac{1}{1+\theta}(y_t - \hat{y}_{t|t-1})$$

令$\alpha = \frac{1}{1+\theta}$，这与SES相同。

对于可逆性（$|\theta| < 1$）：$\alpha \in (0.5, 1)$

### 预测误差方差

对于ARIMA(0,1,1)：
$$\text{Var}(y_{t+h} - \hat{y}_{t+h|t}) = \sigma^2[1 + (h-1)(1-\alpha)^2]$$

预测区间：
$$\hat{y}_{t+h|t} \pm z_{\alpha/2}\sigma\sqrt{1 + (h-1)(1-\alpha)^2}$$

### 最优平滑参数

选择$\alpha$以最小化一步预测误差的平方和：
$$\text{SSE} = \sum_{t=1}^{n}(y_t - \hat{y}_{t|t-1})^2$$

没有闭式解；使用数值优化。

## 算法/模型概述

**SES算法：**

```
输入：时间序列y[1:n]，平滑参数α
输出：预测

1. 初始化：ℓ[0] = y[1]（或前几个值的平均）

2. 对于t = 1到n：
   ℓ[t] = α * y[t] + (1-α) * ℓ[t-1]
   fitted[t] = ℓ[t-1]  # 一步向前

3. 对于h = 1到H：
   forecast[n+h] = ℓ[n]  # 平的预测

返回预测
```

**选择α：**
- $\alpha \to 0$：重度平滑，对变化响应慢
- $\alpha \to 1$：轻度平滑，预测接近最近观测值
- 典型范围：0.1到0.3

## 常见陷阱

1. **对有趋势的数据使用SES**：SES产生平的预测。对于有趋势的数据，使用Holt方法。

2. **对有季节性的数据使用SES**：SES不能捕获季节模式。使用Holt-Winters或先进行季节分解。

3. **任意选择α**：始终使用历史数据优化α或使用交叉验证。

4. **忽视初始化**：$\ell_0$的选择影响早期预测。常见选择：$\ell_0 = y_1$或$\ell_0 = \bar{y}$。

5. **期望预测区间减小**：对于SES，预测区间随步长增长（像随机游走）。

6. **混淆α的解释**：高α = 少平滑（最近数据权重更大）。一些从业者期望相反。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# 生成水平+噪声数据（适合SES）
np.random.seed(42)
n = 100
level = 50
y = level + np.random.randn(n) * 5

# 用优化拟合SES
model = SimpleExpSmoothing(y, initialization_method='estimated')
fit = model.fit(optimized=True)

print(f"最优alpha: {fit.params['smoothing_level']:.3f}")
print(f"初始水平: {fit.params['initial_level']:.2f}")

# 比较不同alpha值
alphas = [0.1, 0.3, 0.5, 0.9]
for alpha in alphas:
    fit_alpha = model.fit(smoothing_level=alpha, optimized=False)
    sse = np.sum((y - fit_alpha.fittedvalues)**2)
    print(f"Alpha={alpha}: SSE={sse:.1f}")

# 预测
forecast = fit.forecast(10)
print(f"\n10步预测（全部相同）: {forecast.values}")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 为什么SES预测对所有未来步长都是常数（平的）？</summary>

<div class="answer">
<strong>答案：</strong> SES将序列建模为局部水平加噪声：$y_t = \ell_t + \epsilon_t$。未来水平的最佳估计是当前水平$\ell_T$。没有建模趋势或季节性，就没有理由预测变化。

<strong>解释：</strong>
预测方程：
$$\hat{y}_{T+h|T} = \ell_T \text{ 对所有 } h \geq 1$$

这假设水平保持不变。不确定性（预测区间）随h增长，但点预测不变。

<div class="pitfall">
<strong>常见陷阱：</strong> 对有趋势的数据使用平的预测导致系统性欠预测/过预测。选择SES前始终检查序列是否有趋势。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> 解释选择平滑参数α时的权衡。</summary>

<div class="answer">
<strong>答案：</strong>
- **高α（接近1）**：最近观测值权重更大。对变化响应快但预测有噪声。适合有频繁水平移动的序列。
- **低α（接近0）**：远期观测值权重更大。预测平滑但适应慢。适合有大量噪声的稳定序列。

**权衡：**
- 响应性vs.稳定性
- 偏差vs.方差（高α = 高方差；低α = 如果水平变化则可能有偏差）

**最优α：** 平衡这些考虑；通过最小化历史数据上的预测误差找到。

<div class="pitfall">
<strong>常见陷阱：</strong> 假设低α总是"更平滑"更好。在波动序列中，低α导致预测滞后于实际水平移动。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 证明SES中的权重和为1。</summary>

<div class="answer">
<strong>答案：</strong> 预测是$\hat{y} = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j y_{t-j}$。

<strong>推导：</strong>
权重和：
$$\sum_{j=0}^{\infty}\alpha(1-\alpha)^j = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j$$

这是公比$(1-\alpha)$的几何级数，其中$|1-\alpha| < 1$：
$$= \alpha \cdot \frac{1}{1-(1-\alpha)} = \alpha \cdot \frac{1}{\alpha} = 1$$

**关键方程：** $\sum_{j=0}^{\infty}r^j = \frac{1}{1-r}$ 当 $|r| < 1$。

<div class="pitfall">
<strong>常见陷阱：</strong> 实际中我们没有无限历史。"缺失的"权重归于初始化$\ell_0$，这就是为什么对短序列初始化很重要。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> 推导SES参数α和ARIMA(0,1,1)参数θ之间的关系。</summary>

<div class="answer">
<strong>答案：</strong> $\alpha = \frac{1}{1+\theta}$ 或等价地 $\theta = \frac{1-\alpha}{\alpha}$。

<strong>推导：</strong>
ARIMA(0,1,1)：$y_t - y_{t-1} = \epsilon_t + \theta\epsilon_{t-1}$

最优h步预测：
$$\hat{y}_{t+1|t} = y_t + \theta\hat{\epsilon}_t$$

其中$\hat{\epsilon}_t = y_t - \hat{y}_{t|t-1}$

这给出：
$$\hat{y}_{t+1|t} = y_t + \theta(y_t - \hat{y}_{t|t-1}) = (1+\theta)y_t - \theta\hat{y}_{t|t-1}$$

重新整理：
$$\hat{y}_{t+1|t} = \frac{1}{1+\theta}(1+\theta)y_t + \frac{\theta}{1+\theta}\hat{y}_{t|t-1}$$

令$\alpha = \frac{1}{1+\theta}$和$1-\alpha = \frac{\theta}{1+\theta}$，这与SES匹配。

<div class="pitfall">
<strong>常见陷阱：</strong> $\theta > 0$的ARIMA(0,1,1)给出$\alpha < 0.5$，这是不可逆的。优化α的标准SES通常给出$\alpha > 0.5$（可逆范围）。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你对月度销售数据应用SES，得到最优α = 0.95。这对你的数据意味着什么？你应该考虑什么？</summary>

<div class="answer">
<strong>答案：</strong> α = 0.95意味着几乎所有权重都在最近观测值上。这表明：

1. 数据**高波动性或频繁水平移动**
2. **可能的趋势**，SES试图通过高响应性来追踪
3. **潜在异常值**将优化拉向高α
4. **近似随机游走行为**

**需要考虑的：**
1. 检查趋势 → 改用Holt方法
2. 寻找异常值 → 它们会使最优α膨胀
3. 绘制序列和拟合值 → 看SES是否在"追逐"数据
4. 尝试Holt或Holt-Winters → 可能用更低的α得到更好的预测
5. 考虑ARIMA(0,1,1) → θ会接近0，确认随机游走

<div class="pitfall">
<strong>常见陷阱：</strong> 很高的α通常表明模型设定错误（缺失趋势或季节性），而不是SES是合适的。接受前先调查。
</div>
</div>
</details>

## 参考文献

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. 第8章。
2. Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing*. Springer.
3. Gardner, E. S. (1985). Exponential smoothing: The state of the art. *Journal of Forecasting*, 4(1), 1-28.
4. Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill.
