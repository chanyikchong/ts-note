# 模型识别

<div class="interview-summary">
<strong>面试摘要：</strong> 模型识别使用ACF/PACF模式、单位根检验和信息准则确定(p,d,q)阶数。AR(p)：PACF在p处截尾。MA(q)：ACF在q处截尾。ARMA：两者都拖尾。使用ADF/KPSS确定d。对于复杂情况，使用AIC/BIC比较候选模型。始终用残差诊断验证。
</div>

## 核心定义

**模型识别**：确定以下内容的过程：
1. 需要的变换（对数、Box-Cox）
2. 差分阶数（d, D）
3. AR阶数（p, P）
4. MA阶数（q, Q）

**纯模型的ACF/PACF模式：**

| 模型 | ACF | PACF |
|-------|-----|------|
| AR(p) | 拖尾（指数/正弦衰减） | 在滞后p处截尾 |
| MA(q) | 在滞后q处截尾 | 拖尾（指数/正弦衰减） |
| ARMA(p,q) | 拖尾 | 拖尾 |
| 非平稳 | 衰减非常慢 | 滞后1处有大尖峰 |
| 白噪声 | 全部接近零 | 全部接近零 |

**显著性阈值**：在白噪声下，$\hat{\rho}(h) \sim N(0, 1/n)$，所以使用$\pm 1.96/\sqrt{n}$作为95%置信带。

## 数学与推导

### 信息准则

**AIC（赤池信息准则）**：
$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

其中$\hat{L}$是最大似然，$k$是参数数量。

对于高斯误差的ARIMA：
$$\text{AIC} = n\ln(\hat{\sigma}^2) + 2(p+q+1)$$

**BIC（贝叶斯/施瓦茨信息准则）**：
$$\text{BIC} = -2\ln(\hat{L}) + k\ln(n)$$

BIC对复杂性的惩罚比AIC更重。

**AICc（修正AIC）**：
$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

小样本首选。

### 扩展ACF（EACF）

EACF方法迭代移除AR结构以识别MA阶数：

1. 拟合AR(0), AR(1), AR(2), ...直到最大值
2. 对每个AR(j)，计算残差的ACF
3. 创建表格：行 = AR阶数，列 = MA阶数
4. "O"表示不显著，"X"表示显著
5. "O"三角形的左上角表明(p,q)

### 单位根检验策略

**ADF + KPSS组合方法：**

| ADF结果 | KPSS结果 | 结论 |
|------------|-------------|------------|
| 拒绝（p < 0.05） | 无法拒绝 | 平稳（d=0） |
| 无法拒绝 | 拒绝 | 非平稳（d≥1） |
| 拒绝 | 拒绝 | 可能有结构性断点 |
| 无法拒绝 | 无法拒绝 | 不确定；尝试差分 |

### 残差自相关的Ljung-Box检验

$$Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$

在零假设（残差为白噪声）下：$Q(m) \sim \chi^2_{m-p-q}$

如果Q很大则拒绝（残差不是白噪声）。

## 算法/模型概述

**完整识别步骤：**

```
步骤1：初步分析
- 绘制序列
- 检查明显的趋势、季节性、异常值
- 如果方差随水平变化则应用变换（对数、平方根）

步骤2：确定差分阶数
- ADF检验：如果p > 0.05，差分
- KPSS检验：如果p < 0.05，差分
- 差分后，重新检验
- 通常d ≤ 2；很少d > 2

步骤3：检查平稳序列的ACF/PACF
- ACF在滞后q处截尾 → 尝试MA(q)
- PACF在滞后p处截尾 → 尝试AR(p)
- 两者都拖尾 → 尝试ARMA

步骤4：拟合候选模型
- 从简单开始：AR(1), MA(1), ARMA(1,1)
- 根据需要增加复杂度
- 比较AIC/BIC

步骤5：诊断检验
- 残差ACF/PACF（应为白噪声）
- Ljung-Box检验
- 残差正态性（Q-Q图）
- 参数显著性

步骤6：选择最终模型
- 在充分模型中选择最低AIC/BIC
- AIC/BIC接近时选择简洁模型
- 良好的残差诊断
```

**自动选择工具：**
- R中的`auto.arima()`（forecast包）
- Python中的`pmdarima.auto_arima()`
- 使用AIC进行逐步搜索

## 常见陷阱

1. **机械地读ACF/PACF**：模式并不总是清晰的。真实数据有噪声。考虑多种解释。

2. **忽视简洁性**：如果ARMA(1,1)和ARMA(2,1)有相似的AIC，优先选择更简单的模型。

3. **过度依赖自动选择**：`auto.arima`是好的起点但可能遗漏重要特征。始终手动验证。

4. **忘记季节模式**：检查季节滞后（月度数据为12, 24, ...）的ACF。标准ARIMA不能捕获这些。

5. **忽视残差诊断**：AIC最低的模型仍可能有自相关残差。始终检查。

6. **样本量问题**：对于小样本（n < 50），ACF/PACF估计不可靠。使用更简单的模型并保守估计。

## 简单示例

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# 生成ARIMA(2,1,1)数据进行识别练习
np.random.seed(42)
n = 250
phi1, phi2, theta = 0.5, -0.2, 0.3
eps = np.random.randn(n + 3)

# 差分序列作为ARMA(2,1)
dX = np.zeros(n)
for t in range(2, n):
    dX[t] = phi1*dX[t-1] + phi2*dX[t-2] + eps[t+1] + theta*eps[t]

X = np.cumsum(dX) + 50  # 积分并添加水平

# 步骤1：检验平稳性
print("步骤1：平稳性检验")
print(f"ADF p值（水平）: {adfuller(X)[1]:.4f}")
print(f"KPSS p值（水平）: {kpss(X, regression='c')[1]:.4f}")

# 步骤2：差分并重新检验
dX_obs = np.diff(X)
print(f"\nADF p值（差分后）: {adfuller(dX_obs)[1]:.4f}")
print(f"KPSS p值（差分后）: {kpss(dX_obs, regression='c')[1]:.4f}")

# 步骤3：差分序列的ACF/PACF
print("\n步骤3：ACF/PACF")
acf_vals = acf(dX_obs, nlags=10)
pacf_vals = pacf(dX_obs, nlags=10)
print(f"ACF: {np.round(acf_vals[1:6], 3)}")
print(f"PACF: {np.round(pacf_vals[1:6], 3)}")

# 步骤4：比较候选模型
print("\n步骤4：模型比较")
models = {
    'ARIMA(1,1,1)': (1,1,1),
    'ARIMA(2,1,0)': (2,1,0),
    'ARIMA(2,1,1)': (2,1,1),
    'ARIMA(1,1,2)': (1,1,2),
}

for name, order in models.items():
    try:
        model = ARIMA(X, order=order).fit()
        lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
        print(f"{name}: AIC={model.aic:.1f}, BIC={model.bic:.1f}, LB p值={lb['lb_pvalue'].values[0]:.3f}")
    except:
        print(f"{name}: 未能收敛")
```

## 测验

<details class="quiz">
<summary><strong>Q1（概念性）：</strong> 为什么我们说MA(q)的ACF"截尾"而AR(p)的ACF"拖尾"？</summary>

<div class="answer">
<strong>答案：</strong>
- MA(q)：$X_t$只涉及$\epsilon_t, \epsilon_{t-1}, \ldots, \epsilon_{t-q}$。超出滞后q，$X_t$和$X_{t-h}$没有共同的$\epsilon$项，所以$\gamma(h) = 0$精确成立。
- AR(p)：$X_t$通过递归结构依赖于所有过去值。虽然只有p个滞后显式出现，依赖链意味着$X_t$与所有$h$的$X_{t-h}$相关。

<strong>关键洞察：</strong> AR有无限记忆（逐渐衰减）；MA有有限记忆（恰好q个滞后）。

<div class="pitfall">
<strong>常见陷阱：</strong> 在实践中，"截尾"不意味着精确为零——由于估计误差，样本ACF在q之后会有小的非零值。寻找急剧下降与逐渐衰减的区别。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2（概念性）：</strong> AIC和BIC有什么区别？什么时候你会偏好其中一个？</summary>

<div class="answer">
<strong>答案：</strong>
- AIC：$-2\ln(L) + 2k$；对参数惩罚轻
- BIC：$-2\ln(L) + k\ln(n)$；惩罚随样本量增长

**偏好BIC的情况：**
- 大样本量（BIC惩罚更合适）
- 当真实模型在候选中时（BIC是一致的）
- 用于推断/解释（更简单的模型）

**偏好AIC的情况：**
- 关注预测（AIC优化预测误差）
- 小样本（使用AICc）
- 当真实模型可能复杂时

<div class="pitfall">
<strong>常见陷阱：</strong> 盲目使用一个准则。对于预测，交叉验证通常比AIC和BIC都好。将信息准则用作指南，而非最终裁决。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3（数学）：</strong> 证明对于白噪声过程，样本ACF的近似方差是$1/n$。</summary>

<div class="answer">
<strong>答案：</strong> 对于$h \neq 0$时$\rho(h) = 0$的白噪声：

<strong>推导（Bartlett近似）：</strong>

对于大n，样本ACF近似正态：
$$\hat{\rho}(h) \sim N(\rho(h), V_h/n)$$

其中$V_h = \sum_{j=-\infty}^{\infty} [\rho(j)\rho(j+h) + \rho(j+h)\rho(j-h) - 2\rho(h)\rho(j)\rho(j+h)]$

对于白噪声（$j \neq 0$时$\rho(j) = 0$）：
$$V_h = 1 \text{ 对所有 } h \neq 0$$

因此：$\text{Var}(\hat{\rho}(h)) \approx 1/n$

**95%置信区间：** $\hat{\rho}(h) \pm 1.96/\sqrt{n}$

<div class="pitfall">
<strong>常见陷阱：</strong> 这个公式只在白噪声下成立。对于非白噪声序列，方差不同（Bartlett公式给出不同的表达式）。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4（数学）：</strong> Ljung-Box检验统计量是$Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$。为什么有$(n-k)$分母？</summary>

<div class="answer">
<strong>答案：</strong> $(n-k)$项是小样本修正。在滞后$k$处，只有$n-k$对观测值贡献于$\hat{\rho}(k)$，使估计精度降低。

<strong>解释：</strong>
原始Box-Pierce统计量使用$Q' = n\sum \hat{\rho}_k^2$。Ljung和Box修改它是因为：
1. $\text{Var}(\hat{\rho}(k)) \approx (n-k)/n^2$而非$1/n$
2. 修正$(n+2)/(n-k)$改善有限样本中的卡方近似

没有修正，检验功效不足（拒绝次数少于应有次数），会遗漏自相关。

<div class="pitfall">
<strong>常见陷阱：</strong> 选择m过大。如果$m > n/4$，检验失去功效。常见选择：非季节数据$m = 10$，季节数据$m = 2s$。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5（实践性）：</strong> 你检查ACF/PACF看到：ACF在前3个滞后慢慢衰减然后下降；PACF在滞后1、2、3有尖峰然后下降。AIC偏好ARIMA(3,0,0)但残差在滞后1处显示显著ACF。你的下一步是什么？</summary>

<div class="answer">
<strong>答案：</strong> 尽管AIC好，滞后1处的显著残差ACF表明模型不充分。尝试：

1. **添加MA(1)**：尝试ARIMA(3,0,1)——MA项可能捕获剩余的滞后1相关性
2. **检查近似冗余**：如果AR(3)系数接近形成MA因子，简化
3. **考虑ARIMA(2,0,1)**：有时混合模型更简洁
4. **检查异常值**：单个异常值可能导致滞后1残差相关
5. **重新检查平稳性**：也许需要$d=1$差分

**关键原则：** 直到残差是白噪声，模型才充分。AIC是必要但非充分——必须通过诊断检验。

<div class="pitfall">
<strong>常见陷阱：</strong> 仅因为AIC最低就接受模型。始终验证残差ACF在所有滞后处都在置信带内，特别是早期滞后。
</div>
</div>
</details>

## 参考文献

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. 第6-8章。
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. 第9章。
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. 第2章。
4. Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.
