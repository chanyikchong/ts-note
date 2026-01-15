# 格兰杰因果检验

<div class="interview-summary">
<strong>面试要点：</strong> 格兰杰因果检验检验X的过去值是否有助于预测Y（超出Y自身历史的预测能力）。这是关于预测先行性，而非真正的因果关系。通过对VAR系数的F检验进行测试。如果X滞后项在Y的方程中联合显著，则X格兰杰因果于Y。可能存在双向因果。对遗漏变量和滞后选择敏感。
</div>

## 核心定义

**格兰杰因果：** 如果满足以下条件，则X格兰杰因果于Y：
$$E[Y_t | Y_{t-1}, Y_{t-2}, \ldots, X_{t-1}, X_{t-2}, \ldots] \neq E[Y_t | Y_{t-1}, Y_{t-2}, \ldots]$$

X的过去提供了超出Y自身过去的关于Y的预测信息。

**非因果性：** 如果知道X的过去不能改善Y的预测，则X不格兰杰因果于Y。

**二元VAR检验：**
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + \sum_{i=1}^{p}\beta_i x_{t-i} + \epsilon_t$$

$H_0$：$\beta_1 = \beta_2 = \cdots = \beta_p = 0$（X不格兰杰因果于Y）

## 数学与推导

### 格兰杰因果的F检验

**受限模型：** 仅Y的AR(p)
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + u_t$$

**非受限模型：** 包含X的VAR
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + \sum_{i=1}^{p}\beta_i x_{t-i} + \epsilon_t$$

**F统计量：**
$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T-2p-1)}$$

在 $H_0$ 下：$F \sim F_{p, T-2p-1}$

### Wald检验（用于VAR）

在VAR框架下，检验：
$$H_0: \mathbf{R}\boldsymbol{\beta} = \mathbf{0}$$

其中 $\mathbf{R}$ 选择Y方程中X滞后项的系数。

Wald统计量：$W = (\mathbf{R}\hat{\boldsymbol{\beta}})'[\mathbf{R}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{R}'\hat{\sigma}^2]^{-1}(\mathbf{R}\hat{\boldsymbol{\beta}})$

在 $H_0$ 下：$W \sim \chi^2_p$

### 瞬时因果

检验当前X是否有助于预测当前Y（超出滞后效应）：

$$\text{Cov}(\epsilon_{yt}, \epsilon_{xt}) \neq 0$$

这检验同期相关，而非时间先行性。

### 块外生性

在多变量系统中，检验一组变量是否格兰杰因果于另一组。

对所有相关系数矩阵进行联合检验。

## 算法/模型框架

**格兰杰因果检验步骤：**

```
1. 确定序列是否平稳
   - 如果是I(1)，差分或使用Toda-Yamamoto方法
   - 标准GC检验要求平稳性

2. 选择最优滞后阶数
   - 对二元VAR使用AIC/BIC
   - 或对所有检验使用相同的p（一致性）

3. 估计非受限VAR(p)

4. 执行Wald/F检验：
   - H0：X滞后项系数 = 0（在Y方程中）
   - 拒绝 → X格兰杰因果于Y

5. 检验反方向：
   - H0：Y滞后项系数 = 0（在X方程中）
   - 拒绝 → Y格兰杰因果于X

6. 解释：
   - 两者都拒绝：双向因果
   - 一个拒绝：单向因果
   - 两者都不拒绝：无格兰杰因果
```

**Toda-Yamamoto方法（用于I(1)序列）：**
1. 确定最大积分阶数 d_max
2. 拟合VAR(p + d_max)
3. 仅检验前p个滞后的系数
4. 避免单位根预检验的问题

## 常见陷阱

1. **与真正因果混淆**：格兰杰因果是预测先行性，不是因果机制。相关可能源于共同原因。

2. **遗漏变量偏差**：如果Z以不同滞后导致X和Y，可能在X和Y之间发现虚假的GC。

3. **错误的滞后选择**：滞后太少 → 遗漏真实效应。太多 → 失去功效并引入噪声。

4. **非平稳数据**：存在单位根时标准F检验的分布是错误的。使用增广滞后方法或误差修正。

5. **多重检验**：检验许多配对会膨胀第一类错误。调整显著性水平。

6. **仅有同期效应**：如果X和Y在同一期内一起变动但不跨期，GC不会检测到。

## 简例

```python
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

# 生成数据：X格兰杰因果于Y，但反之不成立
np.random.seed(42)
n = 200

# X是独立的AR(1)
x = np.zeros(n)
for t in range(1, n):
    x[t] = 0.7 * x[t-1] + np.random.randn()

# Y依赖于自身滞后和X的滞后
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + np.random.randn()

# 堆叠数据
data = np.column_stack([y, x])

# 格兰杰因果检验
print("=== X是否格兰杰因果于Y？ ===")
gc_x_to_y = grangercausalitytests(data, maxlag=4, verbose=True)

print("\n=== Y是否格兰杰因果于X？ ===")
gc_y_to_x = grangercausalitytests(data[:, ::-1], maxlag=4, verbose=True)

# 使用VAR
model = VAR(data)
results = model.fit(2)
print("\n=== 基于VAR的格兰杰因果 ===")
print(results.test_causality('y1', 'y2', kind='f'))  # X → Y
print(results.test_causality('y2', 'y1', kind='f'))  # Y → X
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么格兰杰因果不等同于真正的因果关系？</summary>

<div class="answer">
<strong>答案：</strong> 格兰杰因果仅测量预测先行性——X的过去是否有助于预测Y。它不建立因果机制。

**为什么它们不同：**
1. **遗漏变量**：Z可能以不同滞后导致X和Y，造成虚假的GC
2. **共同原因**：X和Y可能都响应未观测因素
3. **虚假相关**：在独立序列中也可能偶然发现GC
4. **测量时机**：如果X和Y在不同时间测量，GC反映的是测量，而非因果

**例子：** 冰淇淋销量格兰杰因果于溺水（两者都由夏季炎热以不同滞后导致）。

<div class="pitfall">
<strong>常见误区：</strong> 基于GC检验声称X导致Y。始终说"X格兰杰因果于Y"或"X对Y有预测能力"——而不是"X导致Y"。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 双向格兰杰因果意味着什么？这常见吗？</summary>

<div class="answer">
<strong>答案：</strong> 双向GC意味着X → Y和Y → X同时存在（各自有助于预测对方）。这在经济学中很常见。

**例子：**
- GDP ↔ 就业（经济活动和劳动市场相互作用）
- 价格 ↔ 工资（工资-价格螺旋）
- 利率 ↔ 汇率（货币政策和货币市场）

**解释：**
- 反馈关系
- 两个变量都包含独特的预测信息
- 系统是相互依赖的

**注意：** 双向GC不意味着同时因果——它意味着跨时间的相互预测价值。

<div class="pitfall">
<strong>常见误区：</strong> 期望单向因果。在复杂系统中，反馈是常态而非例外。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 对于p个滞后和T个观测的格兰杰因果F检验，自由度是多少？</summary>

<div class="answer">
<strong>答案：</strong> $F_{p, T-2p-1}$（分子自由度 = p，分母自由度 = T - 2p - 1）

**推导：**
- 受限模型：p个参数（p个Y的滞后 + 常数）
- 非受限模型：2p + 1个参数（p个Y的滞后 + p个X的滞后 + 常数）
- 限制：p个参数设为零
- 使用的观测数：T - p（滞后损失p个）

分子自由度 = 限制数 = p
分母自由度 = T - p - (2p + 1) = T - 3p - 1

（某些表述略有不同，取决于是否计算常数。）

**实践中：** 使用软件；这些细节会自动处理。

<div class="pitfall">
<strong>常见误区：</strong> 对于短序列和多滞后，自由度很低，降低检验功效。平衡p与样本量。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> Toda-Yamamoto方法如何处理格兰杰因果检验中的非平稳序列？</summary>

<div class="answer">
<strong>答案：</strong> Toda-Yamamoto（1995）避免单位根预检验：

1. 确定最大积分阶数 $d_{max}$（通常为1或2）
2. 用水平数据拟合VAR(p + $d_{max}$)（不差分）
3. 仅对前p个滞后检验格兰杰因果
4. 额外的 $d_{max}$ 个滞后吸收非平稳性

**为什么有效：**
- 带额外滞后的水平VAR对Wald检验有标准渐近分布
- 无需预检验协整
- 对I(1)或I(0)序列稳健

**检验：**
$$H_0: \beta_1 = \cdots = \beta_p = 0$$

（忽略 $\beta_{p+1}, \ldots, \beta_{p+d_{max}}$）

<div class="pitfall">
<strong>常见误区：</strong> 检验包括额外滞后的所有系数。只检验前p个；额外的 $d_{max}$ 是"干扰"参数。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你使用滞后1-8检验油价和股票收益之间的格兰杰因果。结果变化：在滞后2、4、5处显著，但其他不显著。如何解释？</summary>

<div class="answer">
<strong>答案：</strong> 这种模式表明：

1. **滞后选择很重要**：结果对设定敏感
2. **可能是弱关系**：某些滞后处的显著性可能是偶然的
3. **多重检验**：检验8个设定会膨胀假阳性

**推荐方法：**
1. 首先使用信息准则选择滞后阶数（不是GC结果）
2. 报告最优滞后处的单一检验
3. 如果需要敏感性分析，报告所有结果并承认不稳定性
4. 考虑对多重检验进行Bonferroni校正
5. 在样本外数据上验证

**如果结果不一致：**
- 格兰杰因果的证据弱
- 关系可能是非线性或时变的
- 考虑门限VAR或区制转换模型

<div class="pitfall">
<strong>常见误区：</strong> 选择给出期望结果的滞后（"滞后购物"）。这是p-hacking；报告预先指定的滞后或所有结果。
</div>
</div>
</details>

## 参考文献

1. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.
2. Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66(1-2), 225-250.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第11章.
4. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. 第2章.
