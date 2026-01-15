# 预测区间

<div class="interview-summary">
<strong>面试要点：</strong> 预测区间量化预测不确定性，给出未来值可能落入的范围。对于ARIMA：由于不确定性累积，预测区间宽度随预测步长增大。关键公式：$\hat{y}_{T+h} \pm z_{\alpha/2}\sigma_h$，其中$\sigma_h$取决于模型。区间假设正态性；自助法提供非参数替代方案。
</div>

## 核心定义

**点预测：** 未来值的单一最佳估计
$$\hat{y}_{T+h|T} = E[y_{T+h}|y_1,\ldots,y_T]$$

**预测区间：** 包含未来值的概率为$(1-\alpha)$的范围
$$[\hat{y}_{T+h|T} - z_{\alpha/2}\sigma_h, \hat{y}_{T+h|T} + z_{\alpha/2}\sigma_h]$$

**预测误差：** $e_{T+h|T} = y_{T+h} - \hat{y}_{T+h|T}$

**预测方差：** $\sigma_h^2 = \text{Var}(e_{T+h|T})$

**覆盖概率：** 实际值落入预测区间的比例（应与名义水平一致）。

## 数学推导

### ARMA(p,q) 预测方差

h步预测误差可以写成：
$$e_{T+h|T} = \sum_{j=0}^{h-1}\psi_j\epsilon_{T+h-j}$$

其中$\psi_j$是MA(∞)系数。

预测方差：
$$\sigma_h^2 = \sigma_\epsilon^2\sum_{j=0}^{h-1}\psi_j^2$$

### 具体模型

**AR(1):** $y_t = \phi y_{t-1} + \epsilon_t$
$$\psi_j = \phi^j$$
$$\sigma_h^2 = \sigma_\epsilon^2\frac{1-\phi^{2h}}{1-\phi^2}$$

当 $h \to \infty$: $\sigma_h^2 \to \sigma_\epsilon^2/(1-\phi^2) = \text{Var}(y_t)$

**MA(1):** $y_t = \epsilon_t + \theta\epsilon_{t-1}$
$$\sigma_1^2 = \sigma_\epsilon^2$$
$$\sigma_h^2 = \sigma_\epsilon^2(1+\theta^2) \text{ 当 } h \geq 2$$

**随机游走 (ARIMA(0,1,0)):**
$$\sigma_h^2 = h\sigma_\epsilon^2$$

方差线性增长；预测区间宽度按$\sqrt{h}$增长。

### 高斯预测区间

在正态假设下：
$$y_{T+h}|y_{1:T} \sim N(\hat{y}_{T+h|T}, \sigma_h^2)$$

95% 预测区间: $\hat{y}_{T+h|T} \pm 1.96\sigma_h$
80% 预测区间: $\hat{y}_{T+h|T} \pm 1.28\sigma_h$

### 考虑参数不确定性

当参数是估计得到的，额外的不确定性：
$$\text{Var}(e_{T+h|T}) \approx \sigma_h^2 + \frac{\sigma_h^2}{n}\sum_{j=0}^{h-1}\left(\frac{\partial\psi_j}{\partial\theta}\right)^2\text{Var}(\hat{\theta})$$

对于大样本，参数不确定性相对于内在预测不确定性来说较小。

## 算法/模型概述

**计算预测区间：**

```
1. 拟合模型，估计参数 θ̂ 和 σ̂²
2. 对于每个预测步长 h = 1, ..., H:
   a. 计算点预测 ŷ_{T+h|T}
   b. 计算 ψ₀, ψ₁, ..., ψ_{h-1} (MA系数)
   c. 计算 σ̂ₕ² = σ̂² × Σψⱼ²
   d. 构建区间: ŷ_{T+h|T} ± z_{α/2} × σ̂ₕ

3. 对于非正态数据，使用自助法:
   a. 生成 B 个自助样本
   b. 对每个样本重新拟合模型
   c. 生成预测
   d. 取预测分布的百分位数
```

**自助法预测区间：**
```
对于 b = 1 到 B:
   1. 有放回地抽取残差: ε*[1:n]
   2. 使用模型生成自助序列 y*
   3. 对 y* 重新拟合模型
   4. 生成预测 ŷ*_{T+1:T+H}

取 2.5% 和 97.5% 百分位数 → 95% 预测区间
```

## 常见陷阱

1. **忽视预测区间变宽**：对于I(d)过程，预测区间无界增长。不要期望紧凑的长期预测区间。

2. **假设宽度恒定**：只有平稳AR(∞)过程有有界的预测区间。大多数模型的区间会变宽。

3. **覆盖不足**：如果实际覆盖率 < 名义覆盖率，模型可能设定错误或方差被低估。

4. **覆盖过度**：如果实际覆盖率 >> 名义覆盖率，模型可能过于保守或分布假设错误。

5. **非正态性**：对于偏斜或重尾数据，高斯预测区间可能太窄。使用自助法。

6. **忽视参数不确定性**：小样本 → 参数估计不确定 → 预测区间比公式建议的更宽。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 生成 AR(1) 数据
np.random.seed(42)
phi = 0.7
sigma = 1.0
n = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + np.random.randn() * sigma

# 拟合模型
model = ARIMA(y, order=(1, 0, 0)).fit()

# 获取带预测区间的预测
forecast_obj = model.get_forecast(steps=20)
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% 预测区间

print("带95%预测区间的预测:")
for h in [1, 5, 10, 20]:
    print(f"  h={h}: {forecast.iloc[h-1]:.2f} "
          f"[{conf_int.iloc[h-1, 0]:.2f}, {conf_int.iloc[h-1, 1]:.2f}]")

# AR(1)的理论宽度
phi_hat = model.arparams[0]
sigma_hat = np.sqrt(model.scale)
for h in [1, 5, 10, 20]:
    var_h = sigma_hat**2 * (1 - phi_hat**(2*h)) / (1 - phi_hat**2)
    width_theory = 2 * 1.96 * np.sqrt(var_h)
    width_actual = conf_int.iloc[h-1, 1] - conf_int.iloc[h-1, 0]
    print(f"h={h}: 理论宽度={width_theory:.2f}, 实际宽度={width_actual:.2f}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么大多数时间序列模型的预测区间会随预测步长变宽？</summary>

<div class="answer">
<strong>答案：</strong> 未来的冲击是未知的，并随时间累积。

对于h步预测，我们不知道 $\epsilon_{T+1}, \ldots, \epsilon_{T+h}$。预测误差：
$$e_{T+h|T} = \sum_{j=0}^{h-1}\psi_j\epsilon_{T+h-j}$$

未知冲击越多 → 方差越大 → 区间越宽。

**例外情况：**
- 均值回复过程（平稳AR）收敛到无条件方差
- 但对于随机游走（单位根），方差随h线性增长

<div class="pitfall">
<strong>常见陷阱：</strong> 期望紧凑的长期预测。对于I(1)过程，1年后的预测区间比1天后的宽得多。这是根本性的，不是建模失败。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 如果你的预测区间本应有95%的覆盖率，但实际只有85%，这意味着什么？</summary>

<div class="answer">
<strong>答案：</strong> **覆盖不足** — 实际值落在预测区间之外的频率超出预期。

**可能原因：**
1. **模型设定错误**：未能捕捉真实过程（缺少成分）
2. **非正态性**：重尾导致更多极端值
3. **方差低估**：$\hat{\sigma}$ 太小
4. **结构变化**：模型在稳定期拟合，在波动期测试
5. **忽视参数不确定性**：在小样本中尤其成问题

**补救措施：**
- 使用自助法预测区间
- 检查残差诊断
- 考虑重尾分布
- 手动加宽区间（例如，用99%作为保守的95%）

<div class="pitfall">
<strong>常见陷阱：</strong> 不经验证就信任名义覆盖率。始终在保留数据上检查经验覆盖率。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 推导ARIMA(0,1,0)（随机游走）的h步预测方差。</summary>

<div class="answer">
<strong>答案：</strong> $\sigma_h^2 = h\sigma_\epsilon^2$

**推导：**
随机游走: $y_t = y_{t-1} + \epsilon_t$

h步预测：
$$y_{T+h} = y_T + \sum_{j=1}^{h}\epsilon_{T+j}$$

最佳预测: $\hat{y}_{T+h|T} = y_T$（当前值）

预测误差：
$$e_{T+h|T} = y_{T+h} - y_T = \sum_{j=1}^{h}\epsilon_{T+j}$$

由于 $\epsilon_j$ 独立：
$$\text{Var}(e_{T+h|T}) = \sum_{j=1}^{h}\sigma_\epsilon^2 = h\sigma_\epsilon^2$$

**预测区间:** $y_T \pm 1.96\sigma_\epsilon\sqrt{h}$

宽度按 $\sqrt{h}$ 增长。

<div class="pitfall">
<strong>常见陷阱：</strong> 对于随机游走，长期预测区间会变得非常宽。100步后的95%预测区间是1步后的10倍宽。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 为什么平稳AR(1)在 $h \to \infty$ 时有有界的预测区间宽度？</summary>

<div class="answer">
<strong>答案：</strong> 对于 $|\phi| < 1$ 的平稳AR(1)：

$$\sigma_h^2 = \sigma_\epsilon^2\frac{1-\phi^{2h}}{1-\phi^2}$$

当 $h \to \infty$: $\phi^{2h} \to 0$

$$\lim_{h\to\infty}\sigma_h^2 = \frac{\sigma_\epsilon^2}{1-\phi^2} = \text{Var}(y_t)$$

**直觉：** 对于平稳过程，远期未来值与当前观测独立。预测收敛到无条件均值，不确定性收敛到无条件方差。

预测区间宽度收敛到 $2 \times 1.96 \times \sqrt{\text{Var}(y_t)}$ — 即没有任何数据时你会给出的区间。

<div class="pitfall">
<strong>常见陷阱：</strong> 有界的预测区间不意味着窄区间。无条件方差仍然可能很大，特别是当 $\phi$ 接近1时。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你的模型产生95%预测区间，但利益相关者想知道"最坏情况"。你如何将预测区间转化为业务用途？</summary>

<div class="answer">
<strong>答案：</strong> 几种方法：

1. **使用更高的置信水平**: 99%预测区间给出更保守的边界
   - 99%上界 ≈ 规划的上限

2. **报告具体百分位数**:
   - "95%的可能性需求低于X"
   - "5%的可能性超过Y"

3. **情景分析**:
   - 最佳情况: 80%下界
   - 预期: 点预测
   - 最坏情况: 95%或99%上界

4. **分布摘要**:
   - 最可能范围: 50%预测区间
   - 合理范围: 80%预测区间
   - 极端情景: 95%预测区间

5. **风险分位数**: "损失超过$Z的可能性为10%"

**关键信息：** 预测区间是概率陈述。"最坏情况"取决于可接受的风险水平。

<div class="pitfall">
<strong>常见陷阱：</strong> 将95%上界作为"最坏情况" — 这仍有2.5%的可能被超过。对于真正的尾部风险，使用更高的百分位数或极值方法。
</div>
</div>
</details>

## 参考文献

1. Chatfield, C. (1993). Calculating interval forecasts. *Journal of Business & Economic Statistics*, 11(2), 121-135.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 5.
3. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 5.
4. Thombs, L. A., & Schucany, W. R. (1990). Bootstrap prediction intervals for autoregression. *JASA*, 85(410), 486-492.
