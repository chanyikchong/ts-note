# 信息准则

<div class="interview-summary">
<strong>面试要点：</strong> 信息准则在模型拟合和复杂度之间取得平衡。AIC = -2log(L) + 2k 倾向于预测；BIC = -2log(L) + k·log(n) 倾向于真实模型恢复。越低越好。AIC倾向于选择较大的模型；BIC更加简约。对于小样本，使用AICc。当准则不一致时，考虑你的目标：预测（AIC）vs 推断（BIC）。
</div>

## 核心定义

**AIC（赤池信息准则）：**
$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

**BIC（贝叶斯/施瓦茨信息准则）：**
$$\text{BIC} = -2\ln(\hat{L}) + k\ln(n)$$

**AICc（修正AIC）：**
$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

**组成部分：**
- $\hat{L}$: 最大似然值
- $k$: 估计参数数量
- $n$: 样本量

## 数学推导

### AIC推导（直觉）

AIC最小化真实模型和拟合模型之间的期望Kullback-Leibler散度：
$$\text{KL}(f||g_{\hat{\theta}}) = E_f[\ln f(y)] - E_f[\ln g_{\hat{\theta}}(y)]$$

赤池证明了：
$$E[-2\ln g_{\hat{\theta}}(y_{new})] \approx -2\ln g_{\hat{\theta}}(y) + 2k$$

最小化AIC近似最小化样本外预测误差。

### BIC推导（直觉）

BIC近似对数边际似然：
$$\ln p(y|M) \approx \ln p(y|\hat{\theta},M) - \frac{k}{2}\ln(n)$$

BIC是一致的：如果真实模型在候选模型中，当 n → ∞ 时，BIC选择它的概率 → 1。

### 对于高斯时间序列

残差方差为 $\hat{\sigma}^2$ 时：
$$\text{AIC} = n\ln(\hat{\sigma}^2) + 2k$$
$$\text{BIC} = n\ln(\hat{\sigma}^2) + k\ln(n)$$

### 惩罚项比较

| n | AIC惩罚 | BIC惩罚 |
|---|---------|---------|
| 8 | 2k | 2.08k |
| 20 | 2k | 3.00k |
| 100 | 2k | 4.61k |
| 1000 | 2k | 6.91k |

BIC惩罚随n增长；AIC保持恒定。

## 算法/模型概述

**模型选择流程：**

```
1. 定义候选模型: M₁, M₂, ..., Mₘ
2. 对每个模型 Mᵢ:
   - 用MLE拟合模型
   - 计算AIC和BIC

3. 按准则排序:
   - 用于预测: 优先AIC (或小样本用AICc)
   - 用于推断: 优先BIC

4. 比较顶级候选:
   - ΔAIC < 2: 基本等价
   - ΔAIC 2-7: 对更好模型有一定支持
   - ΔAIC > 10: 对更好模型有强支持

5. 验证:
   - 检查所选模型的残差诊断
   - 考虑样本外测试
```

**赤池权重：**
$$w_i = \frac{\exp(-\frac{1}{2}\Delta\text{AIC}_i)}{\sum_j\exp(-\frac{1}{2}\Delta\text{AIC}_j)}$$

为模型平均提供类似概率的权重。

## 常见陷阱

1. **将准则视为绝对值**：只有相对值重要。AIC = 1000 vs AIC = 1002 是有意义的比较。

2. **忽视样本量对AIC/BIC选择的影响**：对于 n < 40，AICc是必要的。对于大n，BIC可能过于简约。

3. **使用错误的似然**：比较不同变换的模型（对数 vs 水平）需要调整似然。

4. **大样本中AIC过拟合**：随着n增长，AIC允许越来越复杂的模型。考虑BIC以获得简约性。

5. **忽视并列**：如果ΔAIC < 2，模型是等价的。不要过度解释小差异。

6. **忘记模型检验**：最低IC不保证好模型。始终检查残差。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 生成 ARMA(1,1) 数据
np.random.seed(42)
n = 200
phi, theta = 0.7, 0.3
eps = np.random.randn(n + 1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

# 拟合候选模型
candidates = [
    ('AR(1)', (1, 0, 0)),
    ('AR(2)', (2, 0, 0)),
    ('MA(1)', (0, 0, 1)),
    ('MA(2)', (0, 0, 2)),
    ('ARMA(1,1)', (1, 0, 1)),
    ('ARMA(2,1)', (2, 0, 1)),
]

results = []
for name, order in candidates:
    model = ARIMA(y, order=order).fit()
    results.append({
        '模型': name,
        'AIC': model.aic,
        'BIC': model.bic,
        'k': sum(order) + 1  # +1 用于方差
    })

# 按AIC排序显示
import pandas as pd
df = pd.DataFrame(results).sort_values('AIC')
df['ΔAIC'] = df['AIC'] - df['AIC'].min()
df['ΔBIC'] = df['BIC'] - df['BIC'].min()
print(df.to_string(index=False))

# 真实模型 ARMA(1,1) 应该排名靠前
print(f"\nAIC最佳: {df.iloc[0]['模型']}")
print(f"BIC最佳: {df.sort_values('BIC').iloc[0]['模型']}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么BIC倾向于选择比AIC更简单的模型？</summary>

<div class="answer">
<strong>答案：</strong> BIC对参数有更强的惩罚：$k\ln(n)$ vs $2k$。

对于 $n > 8$: $\ln(n) > 2$，所以BIC对每个参数的惩罚更大。

**数学比较：**
- AIC无论样本量都加 $2k$
- BIC加 $k\ln(n)$，随n增长

对于 n = 100: BIC每个参数加4.6k vs AIC的2k。

**后果：** BIC要求更强的似然改进来证明额外参数的合理性，从而导致更简单的模型。

<div class="pitfall">
<strong>常见陷阱：</strong> 认为更简单总是更好。当真实模型复杂时，BIC可能欠拟合。对于预测，AIC通常获胜因为它允许捕捉更多信号。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 什么时候应该使用AICc而不是AIC？</summary>

<div class="answer">
<strong>答案：</strong> 当样本量相对于参数数量较小时使用AICc。

**经验法则：** 当 $n/k < 40$ 时使用AICc。

**为什么用AICc？**
AIC是渐进推导的。对于小样本，它对复杂度的惩罚不足，导致过拟合。

AICc修正项: $\frac{2k(k+1)}{n-k-1}$

当 n ≈ k 时这个额外项很大，但当 n → ∞ 时消失。

**示例：**
- n = 50, k = 5
- AIC惩罚: 10
- AICc惩罚: 10 + 2(5)(6)/(50-6) ≈ 10 + 1.4 = 11.4

<div class="pitfall">
<strong>常见陷阱：</strong> 默认使用AIC而不检查n/k比率。对于小样本，AIC系统性地选择过于复杂的模型。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 从AIC推导AICc的公式。</summary>

<div class="answer">
<strong>答案：</strong> AICc添加了一个偏差修正项：

$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

**推导概要：**
对于高斯误差的回归，Hurvich和Tsai (1989) 证明：

$$E[\text{AIC}] = E[-2\ln L] + 2k$$

当n较小时存在偏差。精确的期望值：

$$E[-2\ln L(\hat{\theta})] + \frac{2kn}{n-k-1}$$

导出：
$$\text{AICc} = -2\ln L + \frac{2kn}{n-k-1} = \text{AIC} + \frac{2k^2 + 2k}{n-k-1}$$

当 $n \to \infty$: $\frac{2k(k+1)}{n-k-1} \to 0$，所以 AICc → AIC。

<div class="pitfall">
<strong>常见陷阱：</strong> AICc公式假设残差方差是估计的。对于受限情况，需要应用不同的修正。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 证明BIC是一致的（选择真实模型）而AIC不是。</summary>

<div class="answer">
<strong>答案：</strong>

**BIC一致性：**
对于嵌套模型，考虑真实模型 $M_0$（k₀个参数）vs 更大的 $M_1$（k₁ > k₀）。

$$\text{BIC}_1 - \text{BIC}_0 = -2(\ln L_1 - \ln L_0) + (k_1 - k_0)\ln n$$

根据似然比理论: $-2(\ln L_1 - \ln L_0) = O_p(1)$（有界）
但惩罚项: $(k_1 - k_0)\ln n \to \infty$

所以 $P(\text{BIC}_1 > \text{BIC}_0) \to 1$。

**AIC不一致：**
$$\text{AIC}_1 - \text{AIC}_0 = -2(\ln L_1 - \ln L_0) + 2(k_1 - k_0)$$

额外参数增加固定惩罚 2(k₁-k₀)，而似然改进是 $O_p(1)$。总存在正概率使额外参数改进拟合到足以抵消固定惩罚。

<div class="pitfall">
<strong>常见陷阱：</strong> 一致性 ≠ 更好的预测。AIC最小化预测误差；BIC识别真实模型。目标不同。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> AIC选择ARIMA(2,1,2)而BIC选择ARIMA(1,1,1)。AIC差异为4。你选择哪个？</summary>

<div class="answer">
<strong>答案：</strong> 取决于你的目标和上下文：

**用于预测：** 倾向于ARIMA(2,1,2)（AIC的选择）
- ΔAIC = 4 表明有意义的预测改进
- 额外复杂度可能捕捉真实动态

**用于解释：** 倾向于ARIMA(1,1,1)（BIC的选择）
- 更简单，更易解释
- 过拟合风险更低

**推荐方法：**
1. 比较样本外预测准确性
2. 检查两者的残差诊断
3. 如果性能相似，优先选择更简单的
4. 考虑集成/平均

**决策矩阵：**

| 因素 | 支持(2,1,2) | 支持(1,1,1) |
|------|-------------|-------------|
| 大样本 | ✓ | |
| 短预测步长 | ✓ | |
| 预期复杂动态 | ✓ | |
| 需要可解释性 | | ✓ |
| 小样本 | | ✓ |
| 长预测步长 | | ✓ |

<div class="pitfall">
<strong>常见陷阱：</strong> 教条地选择一个准则。结合领域知识、验证和判断来使用信息准则。
</div>
</div>
</details>

## 参考文献

1. Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.
2. Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
3. Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference*. Springer.
4. Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297-307.
