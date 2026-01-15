# 时间序列交叉验证

<div class="interview-summary">
<strong>面试要点：</strong> 标准k折交叉验证破坏时间顺序并导致数据泄露。时间序列交叉验证使用滚动/扩展窗口：在过去数据上训练，在未来数据上测试。关键方法：滚动起点、分块CV、h步超前CV。始终尊重时间顺序。CV估计样本外误差；用于模型选择和超参数调优。
</div>

## 核心定义

**时间序列交叉验证：**
通过反复执行以下步骤来评估模型性能：
1. 在过去数据上训练
2. 在未来数据上测试（训练时从未见过）
3. 向前滚动窗口

**滚动起点评估：**
```
训练: [1, ..., t]     → 测试: [t+1, ..., t+h]
训练: [1, ..., t+1]   → 测试: [t+2, ..., t+h+1]
...
训练: [1, ..., T-h]   → 测试: [T-h+1, ..., T]
```

**扩展窗口:** 训练集增长；使用所有历史数据。

**滑动窗口:** 训练集大小固定；丢弃最旧的数据。

## 数学推导

### 滚动起点预测误差

对于起点 $t$ 和预测步长 $h$：
$$e_{t+h|t} = y_{t+h} - \hat{y}_{t+h|t}$$

对所有起点取平均：
$$\text{RMSE}(h) = \sqrt{\frac{1}{T-t_0-h+1}\sum_{t=t_0}^{T-h}e_{t+h|t}^2}$$

### 预测准确性指标

**MAE（平均绝对误差）：**
$$\text{MAE} = \frac{1}{n}\sum|e_t|$$

**RMSE（均方根误差）：**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum e_t^2}$$

**MAPE（平均绝对百分比误差）：**
$$\text{MAPE} = \frac{100}{n}\sum\left|\frac{e_t}{y_t}\right|$$

**SMAPE（对称MAPE）：**
$$\text{sMAPE} = \frac{200}{n}\sum\frac{|e_t|}{|y_t| + |\hat{y}_t|}$$

**MASE（平均绝对标准化误差）：**
$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{n-1}\sum_{t=2}^{n}|y_t - y_{t-1}|}$$

MASE < 1 意味着比朴素预测更好。

### 为什么标准CV失败

标准k折CV：
- 随机划分数据
- 训练折可能包含相对于测试数据的未来观测
- 测试折可能包含过去观测

这导致**数据泄露**：模型在训练期间看到未来信息，给出过于乐观的误差估计。

## 算法/模型概述

**滚动起点CV：**

```python
def rolling_origin_cv(y, model_fn, min_train, horizon, step=1):
    """
    y: 时间序列
    model_fn: 拟合模型并返回预测的函数
    min_train: 最小训练集大小
    horizon: 预测步长
    step: 每次迭代起点移动的步数
    """
    errors = []

    for t in range(min_train, len(y) - horizon, step):
        # 在 [0:t] 上训练，在 [t:t+horizon] 上测试
        train = y[:t]
        test = y[t:t+horizon]

        # 拟合并预测
        forecast = model_fn(train, horizon)

        # 存储误差
        errors.append(test - forecast)

    return np.array(errors)
```

**分块CV（用于相关序列）：**
```
折1: 训练 [块2,3,4,5] → 测试 [块1]
折2: 训练 [块1,3,4,5] → 测试 [块2]
...
```

块是连续的时间段。不太理想但在多个序列共享参数时有用。

## 常见陷阱

1. **使用标准k折CV**：破坏时间顺序，导致泄露。永远不要用于时间序列。

2. **在训练期间测试**：即使使用滚动起点，某些实现也可能意外包含重叠数据。

3. **忽视预测步长**：h=1的CV不保证h=10的良好性能。CV预测步长要与应用匹配。

4. **仅固定起点**：从单一起点测试低估方差。使用多个起点。

5. **计算成本**：完整滚动CV需要重新拟合很昂贵。考虑步长 > 1或固定模型。

6. **非代表性窗口**：如果动态变化，旧数据可能误导。考虑滑动窗口。

## 简单示例

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def mase(actual, forecast, train):
    """平均绝对标准化误差。"""
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(np.diff(train)))
    return mae / naive_mae

# 生成数据
np.random.seed(42)
n = 200
y = np.cumsum(np.random.randn(n)) + 0.1 * np.arange(n)

# 滚动起点CV
min_train = 100
horizon = 5
step = 5

results = {'ARIMA(1,1,0)': [], 'ARIMA(1,1,1)': [], 'ARIMA(2,1,1)': []}

for t in range(min_train, len(y) - horizon, step):
    train = y[:t]
    test = y[t:t+horizon]

    for name, order in [('ARIMA(1,1,0)', (1,1,0)),
                        ('ARIMA(1,1,1)', (1,1,1)),
                        ('ARIMA(2,1,1)', (2,1,1))]:
        try:
            model = ARIMA(train, order=order).fit()
            forecast = model.forecast(horizon)
            error = mase(test, forecast, train)
            results[name].append(error)
        except:
            results[name].append(np.nan)

# 比较模型
print("交叉验证结果 (MASE):")
for name, errors in results.items():
    valid_errors = [e for e in errors if not np.isnan(e)]
    print(f"  {name}: 均值={np.mean(valid_errors):.3f}, "
          f"标准差={np.std(valid_errors):.3f}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么标准k折交叉验证对时间序列失效？</summary>

<div class="answer">
<strong>答案：</strong> 标准k折CV随机将观测分配到折中，破坏了时间顺序。这导致：

1. **数据泄露**：训练数据可能包含相对于测试数据的未来观测
2. **不现实的评估**：实践中，你永远没有未来数据来训练
3. **过于乐观的误差估计**：模型隐式地从未来学习，虚高表观准确性
4. **忽视自相关**：训练和测试中相邻的点是相关的，降低有效测试独立性

**示例：** 如果测试折包含 y[50:60] 而训练包含 y[55:100]，模型在训练期间使用了 y[55:60]（未来！）。

<div class="pitfall">
<strong>常见陷阱：</strong> 直接在时间序列上使用sklearn的 `cross_val_score`。始终使用 `TimeSeriesSplit` 或自定义滚动评估。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 扩展窗口CV和滑动窗口CV有什么区别？</summary>

<div class="answer">
<strong>答案：</strong>

**扩展窗口：**
- 训练集增长: [1:t], [1:t+1], [1:t+2], ...
- 使用所有历史数据
- 对稳定过程更好
- 更多数据 → 更低方差估计

**滑动窗口：**
- 训练集大小固定: [t-w:t], [t-w+1:t+1], ...
- 丢弃最旧的数据
- 对非平稳/演化过程更好
- 适应近期模式

**选择取决于：**
- 平稳性：非平稳 → 滑动
- 数据可用性：有限 → 扩展
- 概念漂移：存在 → 滑动
- 计算成本：滑动更昂贵（总是重新拟合）

<div class="pitfall">
<strong>常见陷阱：</strong> 当动态变化时使用扩展窗口。旧数据误导模型。检查结构断点。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 为什么MASE比MAPE更适合预测评估？</summary>

<div class="answer">
<strong>答案：</strong> MASE的优点：

1. **尺度无关**：像MAPE一样，但能处理零值
2. **无除零问题**：当 $y_t = 0$ 时MAPE失败
3. **对称**：不像MAPE那样不对称地偏向低估/高估
4. **基准比较**：MASE < 1 意味着比朴素预测更好
5. **对间歇性序列有明确定义**：在需求预测中很常见

**公式：**
$$\text{MASE} = \frac{\text{MAE}}{\text{MAE}_{\text{朴素}}}$$

其中MAE_朴素使用季节性朴素或1步朴素作为基准。

**MAPE问题：**
- 当 $y_t = 0$ 时无穷大
- 不对称：y=100时50%的误差（预测50或150）处理方式不同
- 尺度相关的解释

<div class="pitfall">
<strong>常见陷阱：</strong> 对间歇性需求或有零值的数据使用MAPE — 给出未定义或误导性的结果。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 如何为滚动起点CV选择最小训练集大小？</summary>

<div class="answer">
<strong>答案：</strong> 平衡以下因素：

1. **统计要求：**
   - 需要足够数据进行可靠估计
   - 规则：每个参数至少3-5个观测
   - 对于ARIMA(p,d,q)：最少约50 + 10(p+q)个观测

2. **实践考虑：**
   - 更多训练 → 更好的模型估计
   - 但也 → 更少的CV折 → 更高的CV估计方差
   - 典型：数据的60-80%用于第一个训练集

3. **领域知识：**
   - 如果动态变化，近期数据更重要
   - 应包含完整的业务周期（例如，季节性数据要包含全年）

**公式指导：**
$$\text{min\_train} = \max(50, 2 \times m, 5k + 10)$$

其中 m = 季节周期，k = 参数数量。

<div class="pitfall">
<strong>常见陷阱：</strong> 使用太小的min_train导致早期模型不可靠；使用太大则留下太少的CV折来估计方差。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你有3年的日数据（1095个观测），需要为7天预测选择模型。设计一个CV方案。</summary>

<div class="answer">
<strong>答案：</strong>

**推荐方案：**

```
min_train = 365 (1整年以捕捉季节性)
horizon = 7
step = 7 (按周，减少计算量)
```

这产生: (1095 - 365 - 7) / 7 ≈ 103 个CV折

**实现：**
```python
for t in range(365, 1095-7, 7):
    train = data[:t]
    test = data[t:t+7]
    # 拟合并评估
```

**注意事项：**
1. **包含完整季节性**: 365天捕捉年度模式
2. **匹配预测步长**: CV预测步长 = 生产预测步长（7天）
3. **步长 = 预测步长**: 非重叠测试集以保证独立性
4. **指标**: 对每个h=1,...,7使用MASE、MAE、RMSE

**变体：**
- 滑动窗口: 只在最近365天训练（如果非平稳）
- 间隔: 训练/测试之间跳过1-2天以模拟生产延迟

<div class="pitfall">
<strong>常见陷阱：</strong> 对1095个观测使用step=1 → 723次拟合，非常慢。使用更大步长以提高效率。
</div>
</div>
</details>

## 参考文献

1. Bergmeir, C., & Benitez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 5.
3. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF*, 16(4), 437-450.
4. Cerqueira, V., Torgo, L., & Mozetic, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109(11), 1997-2028.
