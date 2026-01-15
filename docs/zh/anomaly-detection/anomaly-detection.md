# 异常检测

<div class="interview-summary">
<strong>面试要点：</strong> 异常检测识别异常观测值——点异常（单个离群值）、集体异常（异常序列）或上下文异常（在特定上下文中异常）。方法包括：统计方法（z分数、IQR）、基于模型的方法（预测残差）、基于距离的方法（LOF）和机器学习方法（隔离森林、自编码器）。主要挑战：定义"正常"和选择阈值。
</div>

## 核心定义

**异常类型：**
- **点异常：** 单个异常值（如突刺）
- **集体异常：** 作为整体异常的序列
- **上下文异常：** 在错误上下文中的正常值（如冬季空调使用量高）

**检测场景：**
- **有监督：** 有标记的正常/异常数据
- **半监督：** 仅使用正常数据进行训练
- **无监督：** 无标签，检测统计离群值

## 数学与推导

### 统计方法

**Z分数：**
$$z_t = \frac{y_t - \bar{y}}{s}$$

如果 $|z_t| > 3$（或选定的阈值）则为异常

**修正Z分数（稳健）：**
$$M_t = \frac{0.6745(y_t - \text{median})}{\text{MAD}}$$

其中 MAD = median(|y - median(y)|)

**IQR方法：**
$$\text{异常条件：} y_t < Q_1 - 1.5\times IQR \text{ 或 } y_t > Q_3 + 1.5\times IQR$$

### 基于模型的检测

拟合时间序列模型，标记大残差：
$$e_t = y_t - \hat{y}_{t|t-1}$$

如果 $|e_t| > k \times \hat{\sigma}$（通常 k = 3）则为异常

**优点：**
- 考虑趋势和季节性
- 适应变化的模式
- 对上下文异常更敏感

### 隔离森林

异常值更容易被隔离（需要更少的分割）。

**算法：**
1. 通过随机分割构建随机树
2. 异常得分 = 隔离点的平均路径长度
3. 路径短 → 异常

**得分：**
$$s(x, n) = 2^{-E[h(x)]/c(n)}$$

其中 h(x) = 路径长度，c(n) = 随机树中的平均路径长度。

### 局部离群因子（LOF）

比较局部密度与邻居的密度：
$$LOF(x) = \frac{\sum_{o \in N_k(x)} \frac{lrd(o)}{lrd(x)}}{|N_k(x)|}$$

LOF >> 1 → 异常（密度低于邻居）

## 算法/模型概述

**时间序列异常检测流程：**

```python
def detect_anomalies(y, method='model', threshold=3):
    if method == 'zscore':
        z = (y - np.mean(y)) / np.std(y)
        return np.abs(z) > threshold

    elif method == 'model':
        # 拟合模型并获取残差
        model = fit_model(y)
        residuals = y - model.fittedvalues
        sigma = np.std(residuals)
        return np.abs(residuals) > threshold * sigma

    elif method == 'rolling':
        # 滚动窗口方法
        window = 30
        rolling_mean = y.rolling(window).mean()
        rolling_std = y.rolling(window).std()
        z = (y - rolling_mean) / rolling_std
        return np.abs(z) > threshold
```

**阈值选择：**
- 固定（z > 3）：简单但可能不适合数据
- 百分位（前1%）：适应分布
- 领域特定：基于假阳性/假阴性的成本
- 极值理论：用于尾部事件

## 常见陷阱

1. **掩蔽效应：** 一个异常值影响均值/标准差，隐藏其他异常。使用稳健统计或基于中位数的方法。

2. **淹没效应：** 正常点因异常值影响而被标记。顺序清理或稳健拟合有帮助。

3. **非平稳性：** 在局部上下文重要时使用全局统计。使用滚动窗口或模型残差。

4. **错误的阈值：** 固定阈值可能过于敏感或过于保守。根据验证数据调整。

5. **忽略季节性：** 周六销售 ≠ 工作日异常。首先建模季节性模式。

6. **集体异常：** 点方法会漏掉异常序列。使用序列感知方法。

## 小型示例

```python
import numpy as np
from scipy import stats

# 生成带有异常的数据
np.random.seed(42)
n = 200
y = np.sin(2 * np.pi * np.arange(n) / 50) + np.random.randn(n) * 0.3

# 插入异常
anomaly_idx = [50, 100, 150]
y[anomaly_idx[0]] += 5   # 正向突刺
y[anomaly_idx[1]] -= 4   # 负向突刺
y[anomaly_idx[2]] += 3   # 较小的突刺

# 方法1：简单z分数
z_scores = np.abs(stats.zscore(y))
detected_zscore = np.where(z_scores > 3)[0]
print(f"Z分数检测到: {detected_zscore}")

# 方法2：滚动z分数
window = 20
rolling_mean = np.convolve(y, np.ones(window)/window, mode='same')
rolling_std = np.array([np.std(y[max(0,i-window):i+1]) for i in range(n)])
rolling_z = np.abs((y - rolling_mean) / rolling_std)
detected_rolling = np.where(rolling_z > 3)[0]
print(f"滚动z分数检测到: {detected_rolling[:10]}...")  # 可能更多

# 方法3：IQR
Q1, Q3 = np.percentile(y, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
detected_iqr = np.where((y < lower) | (y > upper))[0]
print(f"IQR检测到: {detected_iqr}")

# 方法4：基于模型（简单AR）
from statsmodels.tsa.ar_model import AutoReg
model = AutoReg(y, lags=5).fit()
residuals = model.resid
res_z = np.abs(stats.zscore(residuals))
detected_model = np.where(res_z > 3)[0] + 5  # 调整滞后偏移
print(f"基于模型检测到: {detected_model}")

print(f"\n真实异常: {anomaly_idx}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 点异常和上下文异常有什么区别？</summary>

<div class="answer">
<strong>答案：</strong>

**点异常：** 无论上下文如何，值都是异常的。
- 示例：温度读数为500°F
- 检测方法：全局统计、简单阈值

**上下文异常：** 值在某些上下文中正常，在当前上下文中异常。
- 示例：80°F的温度在夏天正常，在冬天异常
- 检测方法：模型残差、条件分布

**为什么这个区别很重要：**
- 点方法（z分数）会漏掉上下文异常
- 季节性/时间模式需要上下文感知方法
- 使用错误方法会导致假阳性

<div class="pitfall">
<strong>常见陷阱：</strong> 在季节性数据上使用简单z分数。12月的值可能对12月来说正常，但与年度均值相比会被标记为异常。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 为什么异常检测中更推荐使用中位数绝对偏差（MAD）而不是标准差？</summary>

<div class="answer">
<strong>答案：</strong> MAD对离群值稳健；标准差则不是。

**标准差的问题：**
单个大离群值 → 膨胀标准差 → 使所有z分数变小 → 离群值"掩蔽"自己和其他值

**MAD的稳健性：**
$$\text{MAD} = \text{median}(|y_i - \text{median}(y)|)$$

- 中位数不受极端值影响
- 50%的数据必须是离群值才能显著影响MAD
- 崩溃点：50% vs 标准差的~0%

**比例因子：**
对于正态数据：$\sigma \approx 1.4826 \times \text{MAD}$

使用修正z分数：$M = \frac{0.6745(y - \text{median})}{\text{MAD}}$

<div class="pitfall">
<strong>常见陷阱：</strong> 当数据可能包含多个离群值时使用基于标准差的z分数。掩蔽效应会隐藏真正的异常。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 推导使用MAD的修正z分数公式。</summary>

<div class="answer">
<strong>答案：</strong> 修正z分数使用MAD而不是标准差进行标准化：

$$M_i = \frac{y_i - \tilde{y}}{\text{MAD}}$$

其中 $\tilde{y}$ = median(y)。

**与标准z分数的比较：**
对于正态分布：$E[\text{MAD}] = \Phi^{-1}(0.75) \times \sigma \approx 0.6745\sigma$

所以：$\sigma \approx \frac{\text{MAD}}{0.6745}$

**修正z分数（缩放后）：**
$$M_i = \frac{0.6745(y_i - \tilde{y})}{\text{MAD}}$$

现在M与正态性假设下的标准z分数具有相同的尺度。
阈值M > 3.5通常使用（等同于|z| > 3）。

<div class="pitfall">
<strong>常见陷阱：</strong> 对修正z分数使用阈值3而没有缩放因子。原始的基于MAD的分数与标准z具有不同的尺度。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 隔离森林如何在不计算距离的情况下检测异常？</summary>

<div class="answer">
<strong>答案：</strong> 隔离森林使用基于树的隔离：

**核心洞察：** 异常值数量少且不同 → 更容易隔离（与其余数据分离）。

**算法：**
1. 随机选择特征和分割值
2. 递归划分数据
3. 异常得分 = 隔离点的平均路径长度

**为什么异常值路径短：**
- 正常点：被相似点包围，需要多次分割
- 异常值：孤立，少数分割就能分离它们

**得分公式：**
$$s(x,n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

- $E[h(x)]$ = x的期望路径长度
- $c(n)$ = n个点的二叉搜索树中的平均路径长度
- $s \to 1$：异常；$s \to 0.5$：正常；$s \to 0$：非常正常

<div class="pitfall">
<strong>常见陷阱：</strong> 隔离森林假设异常值是孤立的。聚集的异常（集体异常）可能被漏掉。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你部署了一个异常检测系统，它每天触发50个警报。调查后发现45个是假阳性。如何改进？</summary>

<div class="answer">
<strong>答案：</strong> 10%的精确率是有问题的。改进方法：

1. **提高阈值：**
   - 将z分数阈值从3提高到4
   - 减少假阳性但可能漏掉真正的异常

2. **添加上下文：**
   - 使用时间、星期几特征
   - 建模季节性模式
   - 仅在上下文内检测上下文异常

3. **集成方法：**
   - 组合多个检测器
   - 仅当多数同意时才标记

4. **从反馈中学习：**
   - 将假阳性标记为正常
   - 重新训练半监督模型

5. **两阶段检测：**
   - 第一阶段：敏感（捕获所有异常）
   - 第二阶段：验证（过滤假阳性）

6. **领域规则：**
   - 添加业务逻辑过滤器
   - 已知的非异常模式

**要跟踪的指标：** 不同阈值下的精确率、召回率、F1分数。

<div class="pitfall">
<strong>常见陷阱：</strong> 仅优化捕获异常（召回率）。高假阳性率导致警报疲劳和警告被忽略。
</div>
</div>
</details>

## 参考文献

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*, 413-422.
3. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *SIGMOD*, 93-104.
4. Hochenbaum, J., Vallis, O. S., & Kejariwal, A. (2017). Automatic anomaly detection in the cloud via statistical learning. *arXiv:1704.07706*.
