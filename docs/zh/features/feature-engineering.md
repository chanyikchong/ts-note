# 时间序列特征工程

<div class="interview-summary">
<strong>面试要点：</strong> 特征工程将原始时间序列转换为机器学习可用的输入。关键特征：滞后项、滚动统计、日期/时间特征、季节性傅里叶项。缩放：标准化或最小-最大缩放，但仅在训练数据上拟合。通过插值或指示变量处理缺失数据。避免泄漏：永远不要在特征中使用未来信息。
</div>

## 核心定义

**滞后特征：** 将过去的值作为预测变量
$$X_{lag_k} = y_{t-k}$$

**滚动特征：** 窗口上的统计量
$$X_{roll\_mean\_k} = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}$$

**日期特征：** 从时间戳提取
- 小时、星期几、月份、季度
- 是否周末、是否节假日
- 距离某事件的天数

**傅里叶特征：** 用正弦/余弦表示季节性模式
$$X_{sin_k} = \sin\left(\frac{2\pi k t}{m}\right), \quad X_{cos_k} = \cos\left(\frac{2\pi k t}{m}\right)$$

## 数学与推导

### 季节性的傅里叶项

对于季节周期m，用K个谐波捕获模式：
$$s(t) = \sum_{k=1}^{K}\left[\alpha_k\sin\left(\frac{2\pi kt}{m}\right) + \beta_k\cos\left(\frac{2\pi kt}{m}\right)\right]$$

**K的选择：**
- K = m/2：捕获所有季节频率
- K = 2-4：通常足以捕获平滑模式
- 使用AIC选择最优K

**为什么使用傅里叶：**
- 处理非整数和长季节周期
- 适用于任何ML模型
- 简洁：2K个特征 vs m个虚拟变量

### 滚动统计

**滚动均值（简单移动平均）：**
$$\bar{y}_t^{(w)} = \frac{1}{w}\sum_{i=0}^{w-1}y_{t-i}$$

**滚动标准差：**
$$s_t^{(w)} = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1}(y_{t-i} - \bar{y}_t^{(w)})^2}$$

**指数移动平均：**
$$\text{EMA}_t = \alpha y_t + (1-\alpha)\text{EMA}_{t-1}$$

### 缩放方法

**标准化（z分数）：**
$$y_{scaled} = \frac{y - \mu_{train}}{\sigma_{train}}$$

**最小-最大缩放：**
$$y_{scaled} = \frac{y - \min_{train}}{\max_{train} - \min_{train}}$$

**稳健缩放：**
$$y_{scaled} = \frac{y - \text{median}_{train}}{\text{IQR}_{train}}$$

**关键：** 始终仅在训练数据上拟合缩放器！

## 算法/模型概述

**特征工程流程：**

```python
def create_features(df, target_col='y', lags=[1,2,3,7],
                   rolling_windows=[7,14,30]):
    features = df.copy()

    # 滞后特征
    for lag in lags:
        features[f'lag_{lag}'] = features[target_col].shift(lag)

    # 滚动特征
    for w in rolling_windows:
        features[f'roll_mean_{w}'] = features[target_col].shift(1).rolling(w).mean()
        features[f'roll_std_{w}'] = features[target_col].shift(1).rolling(w).std()
        features[f'roll_min_{w}'] = features[target_col].shift(1).rolling(w).min()
        features[f'roll_max_{w}'] = features[target_col].shift(1).rolling(w).max()

    # 日期特征（如果是datetime索引）
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    features['is_weekend'] = features.index.dayofweek >= 5

    # 年度季节性的傅里叶特征
    day_of_year = features.index.dayofyear
    for k in range(1, 4):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    return features.dropna()
```

**时间序列的训练-测试分割：**
```python
# 错误：随机分割
X_train, X_test = train_test_split(X)  # 数据泄漏！

# 正确：时间分割
train_end = int(len(X) * 0.8)
X_train, X_test = X[:train_end], X[train_end:]
```

## 常见陷阱

1. **使用未来信息：** 滞后特征必须使用shift(k)，其中k ≥ 1。shift(0) = 泄漏。

2. **在全部数据上缩放：** 仅在训练数据上拟合缩放器。否则测试数据的统计信息会泄漏到训练中。

3. **滚动窗口包含当前值：** 滚动均值应该是`.shift(1).rolling(w)`，而不是`.rolling(w)`。

4. **滞后导致的缺失值：** 创建lag_k后前k个观测值为NaN。删除或填充。

5. **特征过多：** 使用多个滞后和滚动窗口时，维度会爆炸。使用特征选择。

6. **特征的非平稳性：** 如果目标是非平稳的，滞后特征会继承它。考虑差分。

## 小型示例

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 生成示例数据
np.random.seed(42)
n = 365
dates = pd.date_range('2023-01-01', periods=n, freq='D')
trend = np.arange(n) * 0.1
seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 2
y = trend + seasonal + noise

df = pd.DataFrame({'y': y}, index=dates)

# 创建特征
def make_features(df):
    features = df.copy()
    # 滞后
    for lag in [1, 7, 14, 28]:
        features[f'lag_{lag}'] = features['y'].shift(lag)
    # 滚动
    features['roll_mean_7'] = features['y'].shift(1).rolling(7).mean()
    features['roll_std_7'] = features['y'].shift(1).rolling(7).std()
    # 日历
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    # 傅里叶（年度）
    doy = features.index.dayofyear
    features['sin_annual'] = np.sin(2 * np.pi * doy / 365.25)
    features['cos_annual'] = np.cos(2 * np.pi * doy / 365.25)
    return features.dropna()

features = make_features(df)

# 训练-测试分割（时间）
train_size = 300
train = features[:train_size]
test = features[train_size:]

X_train = train.drop('y', axis=1)
y_train = train['y']
X_test = test.drop('y', axis=1)
y_test = test['y']

# 拟合模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"测试RMSE: {rmse:.2f}")

# 特征重要性
importance = pd.Series(model.feature_importances_, index=X_train.columns)
print("\n最重要的特征:")
print(importance.sort_values(ascending=False).head())
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 为什么必须只在训练数据上拟合缩放器？</summary>

<div class="answer">
<strong>答案：</strong> 在全部数据上拟合会导致数据泄漏——测试数据的统计信息会影响训练。

**问题：**
```python
scaler.fit(X)  # 使用了测试数据的统计信息
X_train_scaled = scaler.transform(X_train)  # 训练受测试影响
```

模型在训练期间"知道"了测试数据的范围/分布，导致过于乐观的评估。

**正确方法：**
```python
scaler.fit(X_train)  # 只用训练统计信息
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 应用相同的变换
```

**现实类比：** 在生产中，你没有未来数据来计算统计信息。

<div class="pitfall">
<strong>常见陷阱：</strong> 错误使用sklearn的pipeline。始终在创建pipeline之前分割数据，或在交叉验证中使用TimeSeriesSplit。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 什么时候使用傅里叶特征而不是虚拟变量来表示季节性？</summary>

<div class="answer">
<strong>答案：</strong>

**使用傅里叶当：**
1. **长周期：** 周季节性 = 7个虚拟变量；年季节性 = 365个虚拟变量 vs 2-6个傅里叶项
2. **非整数周期：** 365.25天/年无法用虚拟变量捕获
3. **平滑模式：** 季节性呈正弦形状
4. **线性模型：** 傅里叶项自然捕获周期

**使用虚拟变量当：**
1. **短周期：** 星期几（7个级别）是可管理的
2. **尖锐模式：** "周一效应"是离散的，不是平滑的
3. **可解释性：** 系数直接显示每天的效果
4. **非正弦形状：** 模式不符合正弦/余弦形状

**混合使用：** 可以同时使用——傅里叶用于平滑的年度季节性，虚拟变量用于周度季节性。

<div class="pitfall">
<strong>常见陷阱：</strong> 在日度数据中使用52个虚拟变量表示周度季节性。K=2-4的傅里叶更高效且泛化更好。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 解释为什么shift(1).rolling(w).mean()是正确的而rolling(w).mean()会导致泄漏。</summary>

<div class="answer">
<strong>答案：</strong>

**不使用shift：**
```python
rolling_mean[t] = mean(y[t-w+1], ..., y[t])  # 包含了y[t]！
```
在预测y[t]时，使用rolling_mean[t]包含了y[t]本身 → 泄漏。

**使用shift：**
```python
rolling_mean[t] = mean(y[t-w], ..., y[t-1])  # 不包含y[t]
```
只使用过去的值 → 无泄漏。

**数学符号：**
- 错误：$\bar{y}_t = \frac{1}{w}\sum_{i=0}^{w-1}y_{t-i}$ 包含 $y_t$
- 正确：$\bar{y}_{t-1} = \frac{1}{w}\sum_{i=1}^{w}y_{t-i}$ 不包含 $y_t$

shift将窗口向后移动一个时间步。

<div class="pitfall">
<strong>常见陷阱：</strong> Pandas的rolling默认包含当前值。创建特征时始终在.rolling()前添加.shift(1)。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 需要多少傅里叶项(K)才能完全表示周期为m的季节性？</summary>

<div class="answer">
<strong>答案：</strong> K = m/2项可以完全表示任何周期为m的周期模式。

**解释：**
根据傅里叶定理，任何周期函数都可以表示为：
$$f(t) = \sum_{k=1}^{\infty}\left[a_k\sin\left(\frac{2\pi kt}{m}\right) + b_k\cos\left(\frac{2\pi kt}{m}\right)\right]$$

对于周期为m的离散数据，高于k = m/2的频率会混叠到较低频率（奈奎斯特）。

**实践中：**
- K = m/2：完全表示（2K = m个参数，与虚拟变量相同）
- K = 2-4：通常足够；平滑模式不需要高次谐波
- 使用AIC/BIC：添加项直到没有改进

**示例：** 日度数据中的年度季节性
- 完全：K = 365/2 ≈ 182（过度）
- 典型：K = 3-5（6-10个参数）

<div class="pitfall">
<strong>常见陷阱：</strong> 当K = 3就足够时使用K = m/2。额外的项增加噪声并降低可解释性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 由于滞后窗口，你的滞后特征在开头有30%的缺失值。如何处理？</summary>

<div class="answer">
<strong>答案：</strong> 有几种选择：

1. **删除行（最简单）：**
   ```python
   features = features.dropna()
   ```
   - 丢失初始观测值
   - 如果数据量充足则可行

2. **用第一个可用值填充：**
   ```python
   features = features.fillna(method='bfill')
   ```
   - 使用最早可用的值
   - 轻微偏差但保留数据

3. **使用目标均值/中位数：**
   ```python
   features['lag_7'] = features['lag_7'].fillna(features['y'].mean())
   ```
   - 中性填充
   - 适用于树模型

4. **缺失指示变量：**
   ```python
   features['lag_7_missing'] = features['lag_7'].isna().astype(int)
   features['lag_7'] = features['lag_7'].fillna(0)
   ```
   - 模型学习处理缺失
   - 最灵活

5. **较短的预热滞后：**
   - 开始时使用lag_1，随着数据可用添加更长的滞后
   - 复杂但最大化数据使用

<div class="pitfall">
<strong>常见陷阱：</strong> 当观测值有限时删除30%的数据。先尝试填充；在保留集上验证以检查影响。
</div>
</div>
</details>

## 参考文献

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 7.
2. Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests. *Neurocomputing*, 307, 72-77.
3. Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework for automated time-series phenotyping. *Journal of Open Research Software*, 5(1).
4. Brownlee, J. (2018). *Deep Learning for Time Series Forecasting*. Machine Learning Mastery.
