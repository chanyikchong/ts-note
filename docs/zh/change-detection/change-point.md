# 变点检测

<div class="interview-summary">
<strong>面试要点：</strong> 变点检测识别统计性质（均值、方差、趋势）发生转变的时刻。方法：CUSUM（累积和）、PELT（惩罚精确线性时间）、贝叶斯在线检测。关键权衡：灵敏度与假阳性。应用：过程监控、区制检测、经济学中的结构突变。离线检测使用动态规划；在线检测使用序贯方法。
</div>

## 核心定义

**变点：** 分布参数发生变化的时刻 $\tau$：
$$y_t \sim \begin{cases} F_1(\theta_1) & t < \tau \\ F_2(\theta_2) & t \geq \tau \end{cases}$$

**类型：**
- 均值漂移：$\mu_1 \neq \mu_2$
- 方差变化：$\sigma_1^2 \neq \sigma_2^2$
- 趋势断裂：斜率变化
- 多变点：$\tau_1 < \tau_2 < \cdots < \tau_k$

**场景：**
- 离线：所有数据可用，找出所有变点
- 在线：序贯进行，在变化发生时检测

## 数学与推导

### CUSUM（累积和）

对于已知参数的均值检测：
$$S_t = \sum_{i=1}^{t}(y_i - \mu_0)$$

在H₀下（无变化）：$S_t$ 在0附近波动
在H₁下（在τ处均值漂移）：τ之后 $S_t$ 偏离0

**Page的CUSUM：**
$$C_t^+ = \max(0, C_{t-1}^+ + y_t - \mu_0 - k)$$
$$C_t^- = \max(0, C_{t-1}^- - y_t + \mu_0 - k)$$

当 $C_t^+ > h$ 或 $C_t^- > h$ 时发出变化信号

参数：$k$（容许值），$h$（阈值）

### 二分分割

用于多变点的贪婪算法：

1. 在[1, T]中检验一个变点
2. 如果在τ₁处找到，递归搜索[1, τ₁)和[τ₁, T]
3. 继续直到没有更多显著变化

成本函数（例如，RSS）：
$$C(y_{s:t}) = \sum_{i=s}^{t}(y_i - \bar{y}_{s:t})^2$$

### PELT（惩罚精确线性时间）

通过动态规划的最优分割：
$$F(t) = \min_{s < t}\{F(s) + C(y_{s+1:t}) + \beta\}$$

其中β是每个变点的惩罚。

**剪枝：** 消除次优分割以达到O(n)复杂度。

### 贝叶斯在线变点检测

维护运行长度 $r_t$（自上次变化以来的时间）的概率分布：
$$P(r_t | y_{1:t}) \propto P(y_t | r_t, y_{1:t-1}) P(r_t | r_{t-1})$$

增长概率：$P(r_t = r_{t-1} + 1)$
变化概率：$P(r_t = 0)$

## 算法/模型框架

**离线检测（PELT）：**

```python
def pelt(y, penalty, min_size=2):
    n = len(y)
    F = [0]  # F[t] = y[0:t]的最小成本
    cp = [[]]  # 最优分割的变点

    for t in range(1, n + 1):
        candidates = []
        for s in range(max(0, t - max_segments), t):
            if t - s >= min_size:
                cost = F[s] + segment_cost(y[s:t]) + penalty
                candidates.append((cost, s))

        best_cost, best_s = min(candidates)
        F.append(best_cost)
        cp.append(cp[best_s] + [best_s] if best_s > 0 else [])

    return cp[-1]
```

**在线检测（CUSUM）：**

```python
def cusum_online(y, mu0, k, h):
    n = len(y)
    C_plus, C_minus = 0, 0
    alarms = []

    for t in range(n):
        C_plus = max(0, C_plus + y[t] - mu0 - k)
        C_minus = max(0, C_minus - y[t] + mu0 - k)

        if C_plus > h or C_minus > h:
            alarms.append(t)
            C_plus, C_minus = 0, 0  # 重置

    return alarms
```

## 常见陷阱

1. **惩罚选择**：太小 → 过度分割；太大 → 遗漏变化。使用基于BIC的惩罚或交叉验证。

2. **最小段长度**：非常短的段通常是噪声。强制最小尺寸约束。

3. **多重检验**：检验许多潜在变点会膨胀假阳性。调整阈值。

4. **模型误设**：假设错误分布（例如，对重尾数据假设正态）会影响检测。

5. **渐进vs突然变化**：大多数方法假设突然转变。渐进变化可能表现为多个小变化。

6. **在线延迟**：在线方法检测变化有延迟。速度与准确性之间的权衡。

## 简例

```python
import numpy as np
import ruptures as rpt

# 生成带有2个变点的数据
np.random.seed(42)
n = 300

# 三个不同均值的段
y = np.concatenate([
    np.random.randn(100) + 0,      # 均值 0
    np.random.randn(100) + 3,      # 均值 3
    np.random.randn(100) + 1       # 均值 1
])

# PELT检测
algo = rpt.Pelt(model="l2", min_size=10).fit(y)
change_points = algo.predict(pen=10)
print(f"检测到的变点：{change_points[:-1]}")
print(f"真实变点：[100, 200]")

# 二分分割
algo_binseg = rpt.Binseg(model="l2", min_size=10).fit(y)
change_points_bs = algo_binseg.predict(n_bkps=2)
print(f"BinSeg变点：{change_points_bs[:-1]}")

# 贝叶斯方法（概念性）
# 使用在线贝叶斯检测
from scipy.stats import norm

def bocpd_simple(y, hazard=0.01, mu0=0, sigma=1):
    """简化的贝叶斯在线CPD。"""
    n = len(y)
    R = np.zeros((n + 1, n + 1))  # 运行长度概率
    R[0, 0] = 1

    for t in range(1, n + 1):
        # 每个运行长度下的预测概率
        predprob = norm.pdf(y[t-1], mu0, sigma)  # 简化

        # 增长概率
        R[t, 1:t+1] = R[t-1, :t] * predprob * (1 - hazard)
        # 变化概率
        R[t, 0] = np.sum(R[t-1, :t] * predprob * hazard)
        # 归一化
        R[t, :] /= R[t, :].sum()

    return R

R = bocpd_simple(y, hazard=0.01, mu0=1, sigma=1.5)
print(f"结束时最大概率运行长度：{np.argmax(R[-1])}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 在线与离线变点检测之间的权衡是什么？</summary>

<div class="answer">
<strong>答案：</strong>

**在线检测：**
- 优点：实时警报
- 优点：不需要完整数据
- 缺点：检测延迟（需要积累证据）
- 缺点：无法修正过去决策
- 用例：过程监控、欺诈检测

**离线检测：**
- 优点：使用所有数据进行最优分割
- 优点：可以找到精确的变化位置
- 优点：更准确（全局优化）
- 缺点：非实时
- 用例：历史分析、模型构建

**混合：** 某些方法（如贝叶斯在线）可以离线运行用于回顾分析，同时也提供在线能力。

<div class="pitfall">
<strong>常见误区：</strong> 对离线分析使用在线方法。如果所有数据可用，使用PELT或精确方法以获得更好的准确性。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 如何在PELT或类似方法中选择惩罚参数？</summary>

<div class="answer">
<strong>答案：</strong> 几种方法：

1. **BIC风格惩罚：** $\beta = \log(n) \times k$，其中k = 每段参数数
   - 理论上有依据
   - 可能保守

2. **交叉验证：**
   - 保留数据，选择最小化预测误差的惩罚
   - 计算密集

3. **肘部法则：**
   - 绘制成本vs变点数的图
   - 选择收益递减开始的"肘部"

4. **领域知识：**
   - 预期的变化数量
   - 假阳性vs遗漏检测的代价

**SIC/MBIC：** 为变点特定上下文修改的BIC。

<div class="pitfall">
<strong>常见误区：</strong> 在不同数据集上使用固定惩罚。最优惩罚取决于信噪比、段长度和观测数量。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 推导H₀（无变化）和H₁（均值漂移）下CUSUM的期望值。</summary>

<div class="answer">
<strong>答案：</strong>

**在H₀下：** $y_t \sim N(\mu_0, \sigma^2)$
$$E[S_t] = E\left[\sum_{i=1}^{t}(y_i - \mu_0)\right] = \sum_{i=1}^{t}E[y_i - \mu_0] = 0$$

CUSUM在0附近波动（随机游走行为）。

**在H₁下：** 均值在时刻 $\tau$ 变为 $\mu_1$
$$E[S_t] = \sum_{i=1}^{\tau-1}(\mu_0 - \mu_0) + \sum_{i=\tau}^{t}(\mu_1 - \mu_0) = (t - \tau + 1)(\mu_1 - \mu_0)$$

变化后，CUSUM以 $(\mu_1 - \mu_0)$ 的速率线性偏离0。

**关键洞见：** 漂移率等于均值漂移幅度，使CUSUM对持续变化敏感。

<div class="pitfall">
<strong>常见误区：</strong> CUSUM假设已知的变化前均值 $\mu_0$。如果是估计的，使用标准化CUSUM或调整阈值。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 为什么二分分割不能保证找到最优解？</summary>

<div class="answer">
<strong>答案：</strong> 二分分割是贪婪的——它做出局部最优选择而不考虑全局结构。

**问题场景：**
考虑有两个接近变点的数据：
- 段1：均值0，长度80
- 段2：均值2，长度20
- 段3：均值0，长度100

二分分割首先找到"最佳"单一分割点，可能在位置100附近（段2和段3之间）。然后分别搜索[0,100]和[100,200]。

但小的段2可能通过同时找到两个变点（80和100）来更好地检测。

**解决方案：** 使用考虑所有可能分割的精确方法（PELT、最优分割）。

<div class="pitfall">
<strong>常见误区：</strong> 假设二分分割"足够好"。对于具有不同段大小的复杂信号，精确方法可能显著更好。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你正在监控服务器响应时间。你检测到一个"变点"，但结果发现是单个异常值。如何防止异常值导致的假警报？</summary>

<div class="answer">
<strong>答案：</strong> 几种策略：

1. **稳健成本函数：**
   - 使用L1（绝对值）而非L2（平方）
   - Huber损失
   - 基于中位数的检测

2. **最小段长度：**
   - 要求每段至少k个观测
   - 单个异常值无法形成段

3. **预过滤：**
   - 先应用中位数滤波或异常值去除
   - 然后检测变化

4. **多尺度检测：**
   - 在多个分辨率下检测
   - 真实变化在所有尺度出现；异常值不会

5. **确认期：**
   - 不对第一次偏离发出警报
   - 要求持续变化（例如，适当k的CUSUM）

6. **基于模型：**
   - 使用明确包含异常值成分的模型
   - 将异常值与水平漂移分开

<div class="pitfall">
<strong>常见误区：</strong> 对重尾数据使用平方误差成本。单个大值会主导并触发假变点。
</div>
</div>
</details>

## 参考文献

1. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
2. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA*, 107(500), 1590-1598.
3. Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
4. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.
