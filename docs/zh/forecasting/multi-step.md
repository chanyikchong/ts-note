# 多步预测

<div class="interview-summary">
<strong>面试要点：</strong> 多步预测预测多个未来值。三种策略：递归（迭代1步）、直接（每个预测步长一个单独模型）、MIMO（多输入多输出）。递归累积误差但只用单一模型；直接避免误差累积但需要h个模型。对于ARIMA，递归是标准方法。对于机器学习，通常首选直接方法。
</div>

## 核心定义

**多步预测**: 给定 $y_1, \ldots, y_T$，预测 $y_{T+1}, y_{T+2}, \ldots, y_{T+H}$。

**预测步长 (H)**: 要预测的未来步数。

**策略：**

1. **递归（迭代）**: 重复使用1步模型，将预测值作为输入
2. **直接**: 为每个预测步长h训练单独的模型
3. **MIMO**: 单一模型同时输出所有预测步长
4. **DirRec**: 直接和递归的混合

## 数学推导

### 递归策略

训练模型: $\hat{y}_{t+1} = f(y_t, y_{t-1}, \ldots)$

对于h步预测：
$$\hat{y}_{T+1} = f(y_T, y_{T-1}, \ldots)$$
$$\hat{y}_{T+2} = f(\hat{y}_{T+1}, y_T, \ldots)$$
$$\hat{y}_{T+h} = f(\hat{y}_{T+h-1}, \hat{y}_{T+h-2}, \ldots)$$

**特性：**
- 使用单一模型
- 与底层数据生成过程一致
- 误差通过迭代累积

### 直接策略

训练h个单独的模型：
$$\hat{y}_{t+h}^{(h)} = f_h(y_t, y_{t-1}, \ldots)$$

每个模型直接预测h步之后的值。

**特性：**
- 无误差传播
- 需要h个模型
- 每个模型在不同的目标上训练
- 可能违反跨预测步长的一致性

### 误差分析

**递归误差：**
$$e_{T+h}^{rec} = \sum_{j=1}^{h}\alpha_j\epsilon_{T+j} + O(\text{模型误差})$$

误差通过迭代复合。

**直接误差：**
$$e_{T+h}^{dir} = \epsilon_{T+h}^{(h)} + O(\text{模型误差}_h)$$

无复合，但模型 $f_h$ 可能效率较低。

### 理论比较

**定理（Ben Taieb & Hyndman）：**
在模型正确设定的情况下：
- 递归是最优的（最小化MSFE）
- 直接是一致的但效率较低

在模型设定错误的情况下：
- 直接可能优于递归
- 递归复合设定错误

## 算法/模型概述

**策略选择指南：**

```
如果模型设定正确（ARIMA, ETS）:
   使用 递归
   - 理论最优性
   - 正确的不确定性量化

否则如果使用 机器学习/非参数方法:
   使用 直接
   - 避免误差累积
   - 每个预测步长单独优化

否则如果 预测步长相关:
   使用 MIMO
   - 单一模型，多个输出
   - 可以捕捉预测步长依赖性

对于稳健方法:
   组合 递归和直接
   - 平均预测
   - 通常提高准确性
```

**MIMO 实现：**
```python
# 训练模型一次预测H个步长
# 输入: 特征 X
# 输出: [y_{t+1}, y_{t+2}, ..., y_{t+H}]

X_train, Y_train = create_mimo_data(y, lags=p, horizon=H)
model = MultiOutputRegressor(base_model)
model.fit(X_train, Y_train)

# 预测
X_new = get_features(y[-p:])
forecasts = model.predict(X_new)  # 返回 [ŷ_{T+1}, ..., ŷ_{T+H}]
```

## 常见陷阱

1. **对机器学习使用递归**：基于树的模型外推能力差；递归策略可能产生平坦或爆炸性的预测。

2. **忽视误差累积**：对于长预测步长，递归ARIMA的不确定性增长。不要相信h=100时的紧凑区间。

3. **直接模型不一致**：h=5和h=6的直接模型可能给出 $\hat{y}_{T+6} < \hat{y}_{T+5}$（在预期有趋势时非单调）。

4. **计算成本**：直接方法需要H个模型。对于H=365（日数据，1年），这很昂贵。

5. **不同目标，相同特征**：不同预测步长的直接模型有不同的最优特征。对所有h使用相同特征是次优的。

6. **直接方法中忽视季节性**：月度数据上h=12的直接模型应该捕捉年度模式，但训练数据可能无法提供足够的信号。

## 简单示例

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def create_lagged_data(y, lags, horizon):
    """为直接/MIMO预测创建数据集。"""
    X, Y = [], []
    for t in range(lags, len(y) - horizon):
        X.append(y[t-lags:t][::-1])  # [y_{t-1}, y_{t-2}, ...]
        Y.append(y[t:t+horizon])      # [y_t, y_{t+1}, ...]
    return np.array(X), np.array(Y)

# 生成 AR(2) 数据
np.random.seed(42)
n = 500
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.5 * y[t-1] + 0.3 * y[t-2] + np.random.randn()

# 划分数据
train, test = y[:400], y[400:]
H = 10  # 预测步长

# 递归策略
from statsmodels.tsa.ar_model import AutoReg
model_rec = AutoReg(train, lags=2).fit()
forecast_rec = model_rec.forecast(H)

# 直接策略
X_train, Y_train = create_lagged_data(train, lags=5, horizon=H)
direct_models = [Ridge().fit(X_train, Y_train[:, h]) for h in range(H)]
X_new = train[-5:][::-1].reshape(1, -1)
forecast_dir = np.array([m.predict(X_new)[0] for m in direct_models])

# MIMO策略
mimo_model = MultiOutputRegressor(Ridge()).fit(X_train, Y_train)
forecast_mimo = mimo_model.predict(X_new)[0]

# 比较
print("预测比较:")
print(f"递归:    {np.round(forecast_rec[:5], 2)}")
print(f"直接:    {np.round(forecast_dir[:5], 2)}")
print(f"MIMO:    {np.round(forecast_mimo[:5], 2)}")
print(f"实际:    {np.round(test[:5], 2)}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么递归策略会累积误差而直接策略不会？</summary>

<div class="answer">
<strong>答案：</strong>

**递归：** 每一步都使用预测值作为输入：
$$\hat{y}_{T+2} = f(\hat{y}_{T+1}, y_T, \ldots)$$

$\hat{y}_{T+1}$ 的误差影响 $\hat{y}_{T+2}$，后者又影响 $\hat{y}_{T+3}$，依此类推。

**直接：** 每个预测步长只使用实际观测值：
$$\hat{y}_{T+h} = f_h(y_T, y_{T-1}, \ldots)$$

不同预测步长的误差（在给定数据的情况下）是独立的。

**权衡：**
- 递归：一致但误差复合
- 直接：无复合但效率较低（需要单独模型）

<div class="pitfall">
<strong>常见陷阱：</strong> 认为直接总是更好。对于正确设定的模型，递归是理论最优的。直接主要在设定错误时获胜。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 什么时候你会选择直接而不是递归预测？</summary>

<div class="answer">
<strong>答案：</strong> 在以下情况选择直接：

1. **模型设定错误**：如果1步模型是错误的，递归会复合误差
2. **机器学习方法**：树模型/神经网络在递归模式下通常表现不佳
3. **预测步长特定模式**：不同预测步长有不同的动态
4. **长预测步长**：递归不确定性爆炸；直接保持有界
5. **非平稳特征**：递归可能漂移；直接锚定在数据上

在以下情况选择递归：
- 模型正确设定（ARIMA, ETS）
- 需要一致的概率框架
- 计算效率很重要
- 理解模型动态很重要

<div class="pitfall">
<strong>常见陷阱：</strong> 对梯度提升使用递归 — 树模型不外推，导致长期预测平坦/恒定。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 对于 $y_t = \phi y_{t-1} + \epsilon_t$ 的AR(1)，推导h步递归预测并证明它等于直接最优预测。</summary>

<div class="answer">
<strong>答案：</strong>

**递归：**
$$\hat{y}_{T+1|T} = \phi y_T$$
$$\hat{y}_{T+2|T} = \phi \hat{y}_{T+1|T} = \phi^2 y_T$$
$$\hat{y}_{T+h|T} = \phi^h y_T$$

**直接（最优）：**
条件期望：
$$E[y_{T+h}|y_T] = E[\phi^h y_T + \sum_{j=0}^{h-1}\phi^j\epsilon_{T+h-j}|y_T]$$
$$= \phi^h y_T + 0 = \phi^h y_T$$

它们是相同的！对于正确设定的线性模型，递归 = 直接最优。

**关键见解：** 当模型正确时，用 $\hat{y}_{T+j}$ 代替 $y_{T+j}$ 与直接计算 $E[y_{T+h}|y_{1:T}]$ 给出相同的结果。

<div class="pitfall">
<strong>常见陷阱：</strong> 这种等价性只对线性模型成立。对于非线性模型，即使正确设定，递归 ≠ 直接。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 递归策略和直接策略的预测误差方差有何不同？</summary>

<div class="answer">
<strong>答案：</strong>

**递归（模型正确）：**
$$\text{Var}(e_{T+h}^{rec}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2$$

这是理论最小值（线性预测的Cramér-Rao界）。

**直接（模型正确）：**
$$\text{Var}(e_{T+h}^{dir}) = \text{Var}(e_{T+h}^{rec}) + \text{估计方差}_h$$

直接增加方差是因为模型 $f_h$ 的估计效率低于1步模型（有效使用的数据更少）。

**在设定错误下：**
- 递归: $\text{Var}(e_{T+h}^{rec}) \approx h \times \text{偏差}^2 + \text{方差}$
- 直接: $\text{Var}(e_{T+h}^{dir}) \approx \text{偏差}_h^2 + \text{方差}_h$

直接不会跨预测步长复合偏差。

<div class="pitfall">
<strong>常见陷阱：</strong> 假设直接总是有更大方差。在设定错误下，直接通常获胜因为它避免了复合偏差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你需要预测30天的日销售额。你有一个XGBoost模型。你使用哪种策略？</summary>

<div class="answer">
<strong>答案：</strong> **直接** 或 **MIMO** 策略。

**为什么不用递归：**
- XGBoost是基于树的模型，不能外推
- 递归将预测反馈回去会导致：
  - 预测趋向均值变平
  - 或如果预测漂移到训练范围之外则出现不稳定行为
- 30步递归会显著复合误差

**推荐方法：**
1. **直接：** 训练30个单独的XGBoost模型
   - 针对预测步长优化
   - 可以对每个预测步长使用不同特征
   - 计算成本更高

2. **MIMO：** 训练一个多输出模型
   - 使用 `MultiOutputRegressor(XGBRegressor())`
   - 或自定义多输出架构
   - 比30个模型更高效

3. **混合：** 对短预测步长使用LightGBM/XGBoost，对长预测步长与更简单模型平均

<div class="pitfall">
<strong>常见陷阱：</strong> 使用递归XGBoost — 预测通常退化为常数或振荡。在生产前始终验证多步行为。
</div>
</div>
</details>

## 参考文献

1. Ben Taieb, S., & Hyndman, R. J. (2014). A gradient boosting approach to the Kaggle load forecasting competition. *IJF*, 30(2), 382-394.
2. Chevillon, G. (2007). Direct multi-step estimation and forecasting. *Journal of Economic Surveys*, 21(4), 746-785.
3. Marcellino, M., Stock, J. H., & Watson, M. W. (2006). A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series. *Journal of Econometrics*, 135(1-2), 499-526.
4. Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition. *Expert Systems with Applications*, 39(8), 7067-7083.
