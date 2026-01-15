# ETS框架

<div class="interview-summary">
<strong>面试要点：</strong> ETS（误差-趋势-季节性）是统一指数平滑方法的状态空间框架。以三个组件命名：误差（A/M）、趋势（N/A/Ad/M/Md）、季节性（N/A/M）。提供基于似然的估计和正确的预测区间。ETS(A,A,A) = 加法Holt-Winters。共有30种模型变体。
</div>

## 核心定义

**ETS分类：**

- **E（误差）**：加法(A)或乘法(M)
- **T（趋势）**：无(N)、加法(A)、加法阻尼(Ad)、乘法(M)、乘法阻尼(Md)
- **S（季节性）**：无(N)、加法(A)、乘法(M)

**符号：** ETS(E,T,S)

**示例：**
- ETS(A,N,N) = 简单指数平滑
- ETS(A,A,N) = Holt线性方法
- ETS(A,Ad,N) = 阻尼趋势
- ETS(A,A,A) = 加法Holt-Winters
- ETS(M,A,M) = 乘法误差、加法趋势、乘法季节性

## 数学原理与推导

### 状态空间形式

**一般形式：**
$$y_t = w(\mathbf{x}_{t-1}) + r(\mathbf{x}_{t-1})\epsilon_t$$
$$\mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1})\epsilon_t$$

其中 $\mathbf{x}_t$ 是状态向量（水平、趋势、季节性组件）。

### ETS(A,A,N)：加法误差、加法趋势、无季节性

观测方程：$y_t = \ell_{t-1} + b_{t-1} + \epsilon_t$

状态转移：
$$\ell_t = \ell_{t-1} + b_{t-1} + \alpha\epsilon_t$$
$$b_t = b_{t-1} + \beta\epsilon_t$$

矩阵形式：
$$\begin{pmatrix} \ell_t \\ b_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} \ell_{t-1} \\ b_{t-1} \end{pmatrix} + \begin{pmatrix} \alpha \\ \beta \end{pmatrix}\epsilon_t$$

### ETS(M,A,M)：乘法误差和季节性

观测方程：$y_t = (\ell_{t-1} + b_{t-1})s_{t-m}(1 + \epsilon_t)$

这意味着：$\epsilon_t = \frac{y_t - (\ell_{t-1} + b_{t-1})s_{t-m}}{(\ell_{t-1} + b_{t-1})s_{t-m}}$

状态转移：
$$\ell_t = (\ell_{t-1} + b_{t-1})(1 + \alpha\epsilon_t)$$
$$b_t = b_{t-1} + \beta(\ell_{t-1} + b_{t-1})\epsilon_t$$
$$s_t = s_{t-m}(1 + \gamma\epsilon_t)$$

### 似然函数

对于加法误差：
$$L(\boldsymbol{\theta}|\mathbf{y}) = \prod_{t=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\epsilon_t^2}{2\sigma^2}\right)$$

对于乘法误差：
$$L(\boldsymbol{\theta}|\mathbf{y}) = \prod_{t=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}\mu_t}\exp\left(-\frac{\epsilon_t^2}{2\sigma^2}\right)$$

其中 $\mu_t$ 是一步前预测。

### 预测区间

状态空间表述使得可以进行解析或基于模拟的预测区间估计：

**解析**（对于某些模型）：
$$\text{Var}(y_{T+h}|y_{1:T}) = \sigma^2 \sum_{j=0}^{h-1}c_j^2$$

**模拟**（通用）：
1. 采样未来误差 $\epsilon_{T+1}, \ldots, \epsilon_{T+h}$
2. 使用状态方程生成样本路径
3. 计算预测分布的百分位数

## 算法/模型概述

**ETS模型选择：**

```
1. 考虑所有有效的ETS组合：
   - 共30个模型（某些乘法误差组合不稳定）
   - 稳定模型：根据数据约15-20个

2. 对于每个模型：
   - 通过MLE估计参数
   - 计算AIC/BIC

3. 选择AIC（或BIC）最低的模型

4. 验证：
   - 检查残差诊断
   - 在留出集上比较预测精度

5. 生成带预测区间的预测
```

**有效/稳定模型：**
- 乘法误差需要正值数据
- 某些组合不稳定（如某些参数下的M,Md,M）
- ETS实现通常限制在可容许参数空间

## 常见陷阱

1. **假设ETS = Holt-Winters**：ETS更广泛——包括乘法误差变体并提供适当的统计框架。

2. **忽略乘法误差**：对于方差与水平成比例的正值数据，乘法误差通常拟合更好。

3. **模型平均**：不是选择一个模型，而是对多个ETS模型的预测进行平均可以提高精度。

4. **大季节周期**：$m > 24$ 的ETS通常不实用。改用傅里叶项或TBATS。

5. **负预测**：加法模型可能预测负值。对于正值数据，优先使用乘法组件。

6. **预测区间覆盖率**：检查实际覆盖率是否与标称值匹配（如95%区间应包含约95%的观测值）。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 生成带趋势和乘法季节性的数据
np.random.seed(42)
n = 120
t = np.arange(n)
level = 100 + 0.5 * t
seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
y = level * seasonal * (1 + 0.05 * np.random.randn(n))

# 拟合各种ETS模型
models = {
    'ETS(A,A,A)': {'trend': 'add', 'seasonal': 'add'},
    'ETS(A,A,M)': {'trend': 'add', 'seasonal': 'mul'},
    'ETS(A,Ad,A)': {'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
    'ETS(A,Ad,M)': {'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
}

results = {}
for name, params in models.items():
    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=12,
            **params
        ).fit()
        results[name] = {'AIC': model.aic, 'BIC': model.bic}
    except:
        results[name] = {'AIC': np.inf, 'BIC': np.inf}

print("模型比较:")
for name, metrics in sorted(results.items(), key=lambda x: x[1]['AIC']):
    print(f"  {name}: AIC={metrics['AIC']:.1f}, BIC={metrics['BIC']:.1f}")

# 最佳模型
best_model = min(results.items(), key=lambda x: x[1]['AIC'])[0]
print(f"\n按AIC的最佳模型: {best_model}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念）：</strong> ETS框架相对于传统指数平滑公式的优势是什么？</summary>

<div class="answer">
<strong>答案：</strong> ETS提供：

1. **统计基础**：状态空间表述支持适当的似然推断
2. **模型选择**：AIC/BIC用于系统比较所有30种变体
3. **正确的预测区间**：基于预测误差分布，而非临时公式
4. **统一框架**：所有指数平滑方法在一个一致的符号中
5. **自动选择**：可以通过算法搜索模型

传统公式给出点预测，但缺乏有原则的区间估计和模型比较。

<div class="pitfall">
<strong>常见陷阱：</strong> 使用传统Holt-Winters公式然后用ETS假设计算区间。区间公式取决于误差结构——必须一致。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念）：</strong> 什么时候你会偏好乘法误差(M)而不是加法误差(A)？</summary>

<div class="answer">
<strong>答案：</strong> 当以下情况时使用乘法误差：

1. **方差随水平缩放**：较高的值有较大的绝对误差但相似的百分比误差
2. **仅正值数据**：乘法误差要求 $y_t > 0$
3. **百分比误差有意义**：业务场景中10%误差无论水平如何都类似
4. **异方差性**：方差随时间不恒定

**诊断：** 绘制残差与拟合值。如果方差随拟合值增加 → 乘法误差。

**数学解释：**
- 加法：$y_t = \mu_t + \epsilon_t$（恒定方差）
- 乘法：$y_t = \mu_t(1 + \epsilon_t)$（方差与 $\mu_t^2$ 成比例）

<div class="pitfall">
<strong>常见陷阱：</strong> 对销售/金融数据使用加法误差，而百分比误差是自然的。这低估了高值处的不确定性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学）：</strong> 写出ETS(A,N,N)——简单指数平滑的状态空间方程。</summary>

<div class="answer">
<strong>答案：</strong>

**观测方程：**
$$y_t = \ell_{t-1} + \epsilon_t$$

**状态转移：**
$$\ell_t = \ell_{t-1} + \alpha\epsilon_t$$

**或等价地：**
$$\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$$

这正是SES。状态只是水平 $\ell_t$。预测：$\hat{y}_{t+h|t} = \ell_t$ 对于所有 $h$。

**h步预测误差方差：**
$$\text{Var}(y_{t+h} - \hat{y}_{t+h|t}) = \sigma^2[1 + (h-1)\alpha^2]$$

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记ETS(A,N,N)预测区间随预测步长扩大。平坦的预测不意味着恒定的不确定性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学）：</strong> 为什么某些ETS模型组合是不可容许或不稳定的？</summary>

<div class="answer">
<strong>答案：</strong> 某些组合导致：

1. **负组件**：乘法季节性与加法趋势可能产生 $\ell_t + b_t < 0$，即使季节性为正，也使 $y_t = (\ell_t + b_t) \times s_t$ 为负。

2. **爆炸方差**：某些乘法误差组合的方差随预测步长指数增长。

3. **不可识别**：产生相同预测的参数组合。

**特别有问题的：**
- ETS(M,M,*) — 乘法趋势与乘法误差可能爆炸
- ETS(M,*,M) — 可能给出负预测或无限方差

**可容许区域：** 参数必须满足约束以确保正预测和有界方差。软件强制执行这些约束。

<div class="pitfall">
<strong>常见陷阱：</strong> 手动设置超出可容许边界的参数。始终使用约束优化或让软件处理可容许性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践）：</strong> 你运行自动ETS选择得到ETS(M,Ad,M)，AIC远低于其他选项。在接受这个模型之前应该执行哪些检查？</summary>

<div class="answer">
<strong>答案：</strong> 接受前：

1. **检查残差：**
   - 绘制标准化残差——应该看起来像白噪声
   - 残差ACF——没有显著的尖峰
   - Ljung-Box检验——无法拒绝白噪声

2. **验证假设：**
   - 数据是正值（乘法要求）
   - 方差随水平缩放（证明M误差合理）
   - 季节模式是比例性的（证明M季节性合理）

3. **比较预测：**
   - 如果可能，进行样本外验证
   - 预测看起来合理吗？
   - 检查预测区间覆盖率

4. **参数合理性：**
   - 阻尼参数φ——应该在0.8-0.98
   - α、β、γ——不在边界

5. **与更简单模型比较：**
   - 如果ETS(A,A,M)接近，优先选择更简单的
   - 对数变换 + ETS(A,*,A)可能等价

<div class="pitfall">
<strong>常见陷阱：</strong> 不验证就接受复杂模型。ETS(M,Ad,M)有很多参数——有过拟合风险。始终在留出数据上验证。
</div>
</div>
</details>

## 参考文献

1. Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *IJF*, 18(3), 439-454.
2. Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 8.
4. De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *JASA*, 106(496), 1513-1527.
