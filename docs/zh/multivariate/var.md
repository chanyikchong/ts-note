# 向量自回归（VAR）

<div class="interview-summary">
<strong>面试要点：</strong> VAR联合建模多个时间序列，其中每个变量依赖于自身的滞后值和其他变量的滞后值。VAR(p)：$\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{y}_{t-1} + \cdots + \mathbf{A}_p\mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t$。适用于预测相互关联的序列和分析动态关系。逐方程用OLS估计。用AIC/BIC选择阶数。通过特征值检验稳定性。
</div>

## 核心定义

**VAR(p)模型：**
$$\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{y}_{t-1} + \mathbf{A}_2\mathbf{y}_{t-2} + \cdots + \mathbf{A}_p\mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t$$

其中：
- $\mathbf{y}_t$：k×1变量向量
- $\mathbf{c}$：k×1常数向量
- $\mathbf{A}_i$：k×k系数矩阵
- $\boldsymbol{\epsilon}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})$：k×1误差向量

**紧凑形式：**
$$\mathbf{A}(L)\mathbf{y}_t = \mathbf{c} + \boldsymbol{\epsilon}_t$$

其中 $\mathbf{A}(L) = \mathbf{I} - \mathbf{A}_1 L - \cdots - \mathbf{A}_p L^p$

**平稳性条件：** 伴随矩阵的所有特征值在单位圆内。

## 数学与推导

### 二元VAR(1)示例

对于变量 $(y_{1t}, y_{2t})$：
$$\begin{pmatrix} y_{1t} \\ y_{2t} \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}\begin{pmatrix} y_{1,t-1} \\ y_{2,t-1} \end{pmatrix} + \begin{pmatrix} \epsilon_{1t} \\ \epsilon_{2t} \end{pmatrix}$$

展开写法：
$$y_{1t} = c_1 + a_{11}y_{1,t-1} + a_{12}y_{2,t-1} + \epsilon_{1t}$$
$$y_{2t} = c_2 + a_{21}y_{1,t-1} + a_{22}y_{2,t-1} + \epsilon_{2t}$$

交叉系数 $a_{12}, a_{21}$ 捕捉动态溢出效应。

### 伴随形式

VAR(p)可以在更高维度写成VAR(1)：
$$\boldsymbol{\xi}_t = \mathbf{F}\boldsymbol{\xi}_{t-1} + \mathbf{v}_t$$

其中 $\boldsymbol{\xi}_t = (\mathbf{y}_t', \mathbf{y}_{t-1}', \ldots, \mathbf{y}_{t-p+1}')'$，$\mathbf{F}$ 是伴随矩阵。

平稳性：$\mathbf{F}$ 的特征值模小于1。

### MA(∞)表示

平稳VAR具有移动平均形式：
$$\mathbf{y}_t = \boldsymbol{\mu} + \sum_{i=0}^{\infty}\boldsymbol{\Phi}_i\boldsymbol{\epsilon}_{t-i}$$

$\boldsymbol{\Phi}_i$ 是脉冲响应矩阵：$\boldsymbol{\Phi}_i^{jk}$ = 变量j对变量k在滞后i处冲击的响应。

### 预测误差方差分解

变量j的h步预测误差方差：
$$\sigma_j^2(h) = \sum_{i=0}^{h-1}\sum_{k=1}^{K}(\Phi_i^{jk})^2\sigma_k^2$$

变量k对变量j在期限h的方差贡献。

## 算法/模型框架

**VAR估计：**

```
1. 确定最优滞后阶数p：
   - 拟合 VAR(1), VAR(2), ..., VAR(p_max)
   - 选择最小化AIC或BIC的p

2. 用OLS估计：
   - 每个方程可以单独估计
   - OLS是一致且有效的（相同回归变量）

3. 检查稳定性：
   - 计算伴随矩阵的特征值
   - 所有 |λᵢ| < 1 为平稳

4. 诊断：
   - 检验残差自相关（多元LB检验）
   - 检验正态性
   - 检验异方差性

5. 分析：
   - 脉冲响应
   - 预测误差方差分解
   - 格兰杰因果检验
```

## 常见陷阱

1. **参数太多**：具有k个变量的VAR(p)有 k + k²p 个参数。高k或p容易过拟合。

2. **非平稳变量**：VAR要求平稳性。对I(1)变量使用差分或VECM。

3. **结构解释**：简约形式VAR显示相关性，而非因果关系。对于因果声明使用结构VAR(SVAR)。

4. **忽略协整**：如果变量是协整的，受限VECM比无限制的差分VAR更有效。

5. **过度解读IRF**：脉冲响应取决于排序（Cholesky）或识别假设。

6. **忘记同期相关**：$\boldsymbol{\Sigma}$ 不是对角矩阵；冲击跨方程相关。

## 简例

```python
import numpy as np
from statsmodels.tsa.api import VAR

# 生成二元VAR(1)数据
np.random.seed(42)
n = 200
A = np.array([[0.5, 0.3],
              [0.2, 0.4]])
c = np.array([1, 2])

y = np.zeros((n, 2))
for t in range(1, n):
    y[t] = c + A @ y[t-1] + np.random.randn(2) * 0.5

# 拟合VAR
model = VAR(y)

# 选择滞后阶数
lag_order = model.select_order(maxlags=8)
print("滞后阶数选择：")
print(lag_order.summary())

# 拟合VAR(1)
results = model.fit(1)
print("\n系数矩阵A：")
print(results.coefs[0])
print(f"\n真实A：\n{A}")

# 脉冲响应
irf = results.irf(10)
print(f"\nIRF：y1对y2冲击在滞后5的响应：{irf.irfs[5, 0, 1]:.3f}")

# 预测
forecast = results.forecast(y[-1:], steps=5)
print(f"\n5步预测：\n{forecast}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> VAR相比分别拟合单变量模型有什么优势？</summary>

<div class="answer">
<strong>答案：</strong> VAR捕捉跨变量动态：

1. **动态交互**：$y_1$ 如何影响未来的 $y_2$，反之亦然
2. **联合预测**：使用所有变量的信息
3. **相关误差**：考虑同期冲击
4. **政策分析**：脉冲响应显示系统范围的影响

**例子：** GDP和通货膨胀。VAR捕捉：
- 过去GDP影响未来通胀（需求效应）
- 过去通胀影响未来GDP（实际余额效应）
- 同时影响两者的相关供给冲击

分别的ARIMA会遗漏这些交互。

<div class="pitfall">
<strong>常见误区：</strong> 当变量不相关时使用VAR。有k个变量，每个滞后要估计k²个系数——如果许多为零则浪费。考虑稀疏VAR或变量选择。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 为什么变量的排序对脉冲响应分析很重要？</summary>

<div class="answer">
<strong>答案：</strong> 标准IRF使用 $\boldsymbol{\Sigma}$ 的Cholesky分解，这取决于变量排序。

**Cholesky：** $\boldsymbol{\Sigma} = \mathbf{PP}'$，其中 $\mathbf{P}$ 是下三角矩阵。

这意味着：
- 第一个变量的冲击是"结构性的"（同期不受其他变量影响）
- 后面的变量在同一期内响应前面的变量

**不同排序 → 不同IRF**

**例子：** 排序 (GDP, 通胀) vs (通胀, GDP)
- 第一种排序：GDP冲击立即影响通胀
- 第二种排序：通胀冲击立即影响GDP

**解决方案：**
- 用理论来证明排序合理
- 使用具有明确识别的结构VAR
- 报告对排序的敏感性

<div class="pitfall">
<strong>常见误区：</strong> 报告IRF而不说明排序或证明识别。结果可能由任意的排序选择驱动。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 对于系数矩阵为 $\mathbf{A}$ 的VAR(1)，平稳性条件是什么？</summary>

<div class="answer">
<strong>答案：</strong> $\mathbf{A}$ 的所有特征值的模必须小于1。

**原因：**
VAR(1)：$\mathbf{y}_t = \mathbf{c} + \mathbf{A}\mathbf{y}_{t-1} + \boldsymbol{\epsilon}_t$

向后迭代：
$$\mathbf{y}_t = (\mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \cdots)\mathbf{c} + \sum_{j=0}^{\infty}\mathbf{A}^j\boldsymbol{\epsilon}_{t-j}$$

这收敛当且仅当 $\mathbf{A}^j \to 0$，这要求所有特征值在单位圆内。

**对于二元情形：**
如果 $\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$：

特征值：$\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}$

需要 $|\lambda_1| < 1$ 且 $|\lambda_2| < 1$。

<div class="pitfall">
<strong>常见误区：</strong> 只检查对角元素。即使 $|a|, |d| < 1$，非对角项也可能使系统不稳定。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 具有k个变量的VAR(p)模型有多少参数？</summary>

<div class="answer">
<strong>答案：</strong> $k + k^2 p + \frac{k(k+1)}{2}$

**分解：**
- $k$ 个常数项（向量 $\mathbf{c}$）
- $k^2 \times p$ 个系数（p个大小为k×k的矩阵）
- $\frac{k(k+1)}{2}$ 个方差-协方差参数（对称 $\boldsymbol{\Sigma}$）

**例子：** k=3个变量，p=4个滞后：
- 常数：3
- AR系数：9 × 4 = 36
- 协方差：6
- 总计：45个参数

**含义：**
- 参数以 $k^2$ 增长
- 数据有限时，过拟合严重
- 考虑受限VAR、BVAR或变量选择

<div class="pitfall">
<strong>常见误区：</strong> 用小样本拟合大型VAR。经验法则：每个参数至少需要10-20个观测。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你对3个季度宏观变量拟合VAR(2)。残差自相关检验在滞后4处拒绝。你该怎么做？</summary>

<div class="answer">
<strong>答案：</strong> 季度数据滞后4的自相关表明年度季节性。选项：

1. **增加滞后阶数**：尝试VAR(4)或VAR(5)以捕捉年度动态

2. **添加季节虚拟变量**：将Q1、Q2、Q3指标作为外生变量

3. **季节调整**：预先过滤数据以去除季节性

4. **VARX**：添加季节傅里叶项作为外生回归变量

**诊断过程：**
1. 检查是否所有三个残差都显示滞后4的模式
2. 拟合VAR(4)并重新检验
3. 比较AIC：带季节项的VAR(2) vs VAR(4)
4. 验证残差现在通过检验

**考虑因素：** 更多滞后 = 更多参数。如果样本小，优先选择季节虚拟变量而非VAR(4)。

<div class="pitfall">
<strong>常见误区：</strong> 忽略宏观数据中的季节模式。年度效应很常见；季度VAR应明确捕捉它们。
</div>
</div>
</details>

## 参考文献

1. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第10-11章.
3. Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
4. Stock, J. H., & Watson, M. W. (2001). Vector autoregressions. *Journal of Economic Perspectives*, 15(4), 101-115.
