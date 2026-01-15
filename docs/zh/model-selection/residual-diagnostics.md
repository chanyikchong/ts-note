# 残差诊断

<div class="interview-summary">
<strong>面试要点：</strong> 如果模型合适，残差应该是白噪声。通过以下方式检验：ACF图（无显著尖峰）、Ljung-Box检验（p > 0.05）、正态性（Q-Q图）和同方差性（恒定方差）。残差中的模式表明模型不充分：自相关提示缺少AR/MA项；方差变化提示GARCH；趋势提示差分不当。
</div>

## 核心定义

**残差：** $e_t = y_t - \hat{y}_{t|t-1}$（一步超前预测误差）

**标准化残差：** $z_t = e_t / \hat{\sigma}$（应近似为N(0,1)）

**白噪声性质：**
- $E[e_t] = 0$
- $\text{Var}(e_t) = \sigma^2$（恒定）
- $\text{Cov}(e_t, e_{t-k}) = 0$ 对于 $k \neq 0$

## 数学推导

### Ljung-Box检验

检验自相关是否联合为零。

$$Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$

在H₀（白噪声）下：$Q(m) \sim \chi^2_{m-p-q}$（根据估计参数调整）

**决策：** 如果 Q > 临界值（或 p < α）则拒绝H₀

### Jarque-Bera正态性检验

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

其中 S = 偏度，K = 峰度。

在H₀（正态性）下：$JB \sim \chi^2_2$

### ARCH-LM异方差检验

检验 $e_t^2$ 是否依赖于过去的平方残差：
$$e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + \cdots + \alpha_p e_{t-p}^2 + v_t$$

检验统计量：$nR^2 \sim \chi^2_p$ 在H₀（同方差）下

### 游程检验随机性

计算游程（连续同号残差）。游程太少提示自相关；游程太多提示过度差分。

## 算法/模型概述

**诊断检查清单：**

```
1. 均值为零
   □ 残差均值 ≈ 0
   □ 残差随时间绘图：无趋势

2. 无自相关
   □ ACF图：所有尖峰在 ±1.96/√n 带内
   □ PACF图：无模式
   □ Ljung-Box检验：多个滞后处 p > 0.05

3. 恒定方差
   □ 残差vs时间图：无扇形/聚集
   □ 残差vs拟合值图：无模式
   □ ARCH检验：p > 0.05

4. 正态性（不太关键）
   □ 直方图：大致钟形
   □ Q-Q图：点在对角线上
   □ Jarque-Bera：p > 0.05

5. 无异常值
   □ |标准化残差| < 3 大部分情况
   □ 检查任何 > 3 的点是否有数据问题
```

**违反的解释：**

| 违反 | 解释 | 修复 |
|------|------|------|
| 滞后1 ACF尖峰 | 缺少MA(1) | 添加MA项 |
| 滞后1 PACF尖峰 | 缺少AR(1) | 添加AR项 |
| 季节性尖峰 | 缺少季节性 | 添加季节性项 |
| ACF缓慢衰减 | 差分不足 | 增加d |
| 滞后1负ACF | 过度差分 | 减少d |
| 方差变化 | 异方差性 | GARCH，对数变换 |
| 非正态 | 重尾 | 稳健方法，异常值处理 |

## 常见陷阱

1. **过度检验**：有许多滞后时，有些会偶然显著。关注早期滞后和模式。

2. **忽视自由度**：Ljung-Box的df = m - p - q，不是m。错误的df给出错误的p值。

3. **m选择不当**：m太小会遗漏长程依赖；太大则功效低。规则：m ≈ min(10, n/5)。

4. **正态性执念**：非正态性通常是可接受的。自相关是关键检查。

5. **遗漏季节性滞后的模式**：始终检查滞后12、24（月度）、7、14（日度）等的ACF。

6. **混淆残差和新息**：对于MA模型，残差 ≠ 真实新息。有限样本中预期有一些自相关。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

# 生成数据并拟合故意错误的模型
np.random.seed(42)
n = 200
# 真实模型: ARMA(1,1)
phi, theta = 0.7, 0.4
eps = np.random.randn(n + 1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

# 只拟合AR(1)（缺少MA项）
model_wrong = ARIMA(y, order=(1, 0, 0)).fit()
resid = model_wrong.resid

print("=== 残差诊断 ===\n")

# 1. 均值
print(f"1. 均值: {np.mean(resid):.4f} (应该 ≈ 0)")

# 2. 自相关
print("\n2. 自相关:")
lb_test = acorr_ljungbox(resid, lags=[5, 10, 15], return_df=True)
print(lb_test)

# 3. 正态性
jb_stat, jb_p = stats.jarque_bera(resid)
print(f"\n3. 正态性 (Jarque-Bera): stat={jb_stat:.2f}, p={jb_p:.4f}")

# 4. 检查ACF
acf_vals = np.correlate(resid, resid, mode='full')
acf_vals = acf_vals[len(acf_vals)//2:] / acf_vals[len(acf_vals)//2]
print(f"\n4. 滞后1的ACF: {acf_vals[1]:.3f} (如果|.| > {1.96/np.sqrt(n):.3f}则显著)")

# 与正确模型比较
model_correct = ARIMA(y, order=(1, 0, 1)).fit()
resid_correct = model_correct.resid
lb_correct = acorr_ljungbox(resid_correct, lags=[5, 10, 15], return_df=True)
print("\n=== 正确模型 (ARMA(1,1)) ===")
print(lb_correct)
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么检查残差自相关比检查正态性更重要？</summary>

<div class="answer">
<strong>答案：</strong>

**自相关更重要因为：**
1. **有偏预测**：残差自相关意味着系统性模式未被利用
2. **无效推断**：标准误差和置信区间假设独立性
3. **模型不充分**：自相关直接表明缺少结构
4. **可修复**：可以添加AR/MA项来消除自相关

**正态性不太关键因为：**
1. **存在稳健方法**：点预测不需要正态性
2. **中心极限定理帮助**：即使残差不正态，平均值也趋向正态
3. **只影响区间**：正态性主要影响预测区间
4. **通常可忽略**：重尾不会使预测有偏，只是加宽区间

<div class="pitfall">
<strong>常见陷阱：</strong> 花精力在正态性变换上而忽视自相关。先修复自相关；正态性通常可以忽略。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 残差ACF在滞后1处有显著负尖峰意味着什么？</summary>

<div class="answer">
<strong>答案：</strong> 可能是**过度差分**。

**解释：**
对平稳序列差分会引入近似θ ≈ -1的MA(1)：
$$(1-L)y_t = \epsilon_t - \epsilon_{t-1} \text{ 大致}$$

其ACF: $\rho(1) = -1/(1+1) = -0.5$

所以大的负滞后1 ACF（约-0.3到-0.5）提示你对已经平稳的序列进行了差分。

**行动：**
1. 重新检验原始序列的平稳性
2. 尝试不差分的模型
3. 比较d=0和d=1的AIC

<div class="pitfall">
<strong>常见陷阱：</strong> 反射性地差分因为这是"标准程序"。在差分前后检查平稳性检验和残差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> Ljung-Box检验使用自由度m-p-q。为什么要减去p+q？</summary>

<div class="answer">
<strong>答案：</strong> 我们减去估计参数数量以考虑它们对残差的影响。

**解释：**
对于拟合的ARMA(p,q)，残差 $e_t = y_t - \hat{y}_{t|t-1}$ 是使用估计的 $\hat{\phi}$, $\hat{\theta}$ 计算的。

估计过程使用了数据中的信息，降低了有效自由度。具体来说：
- p个AR参数约束了p个滞后自相关
- q个MA参数约束了q个滞后自相关

在H₀下，检验统计量：
$$Q(m) \sim \chi^2_{m-p-q}$$

而不是 $\chi^2_m$。使用m个自由度会导致拒绝过于频繁（检验过大）。

<div class="pitfall">
<strong>常见陷阱：</strong> 软件可能不会自动调整df。验证p和q被减去；否则p值是错误的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 如何解释残差的ARCH-LM检验？</summary>

<div class="answer">
<strong>答案：</strong> ARCH-LM检验条件异方差性 — 方差是否依赖于过去的波动。

**过程：**
1. 计算平方残差 $e_t^2$
2. 回归: $e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + \cdots + \alpha_p e_{t-p}^2$
3. 检验: $H_0$: 所有 $\alpha_i = 0$（同方差）

**检验统计量：** $nR^2 \sim \chi^2_p$

**解释：**
- p < 0.05: 有ARCH效应的证据；方差聚集
- p > 0.05: 无证据；恒定方差假设成立

**如果显著：**
- 考虑GARCH模型
- 或方差稳定变换（对数）
- 预测区间需要调整

<div class="pitfall">
<strong>常见陷阱：</strong> 忽视ARCH效应导致预测区间在波动期太窄，在平静期太宽。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你拟合了ARIMA(1,1,1)，残差诊断显示：滞后10处Ljung-Box p=0.02，但滞后5和15处p=0.15。Q-Q图显示轻微重尾。你得出什么结论？</summary>

<div class="answer">
<strong>答案：</strong> 模型可能是充分的；不要过度解释滞后10的结果。

**分析：**
1. **滞后10处Ljung-Box：** p=0.02是边缘的。但滞后5和15正常。
   - 可能是虚假的（多重检验）
   - 或者是对预测不重要的轻微模型不足

2. **重尾：** 在经济/金融数据中常见
   - 不使预测无效
   - 影响预测区间（可能需要更宽）

**推荐行动：**
1. 视觉检查ACF — 滞后10处的孤立尖峰可能是噪声
2. 与更简单模型比较（ARIMA(1,1,0)）— 如果预测相似，优先选择更简单的
3. 对区间，考虑自助法或t分布
4. 在保留数据上验证 — 最终检验

**结论：** 接受模型除非保留验证显示问题。完美的残差是不现实的；"足够好"是标准。

<div class="pitfall">
<strong>常见陷阱：</strong> 追求完美诊断。添加参数来修复一个边缘检验通常导致过拟合。关注预测性能。
</div>
</div>
</details>

## 参考文献

1. Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 8.
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. Chapter 2.
4. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
