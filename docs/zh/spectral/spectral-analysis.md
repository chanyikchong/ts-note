# 谱分析

<div class="interview-summary">
<strong>面试要点：</strong> 谱分析将时间序列分解为频率成分。周期图估计每个频率上的功率。峰值表示主要周期。对于平稳序列，谱密度是自协方差的傅里叶变换。关键洞见：AR产生平滑谱；MA产生峰值。适用于检测隐藏的周期性和理解周期行为。
</div>

## 核心定义

**谱密度：** 对于平稳过程，谱密度 $f(\omega)$ 表示频率 $\omega$ 处的功率：
$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$

**周期图：** 谱密度的样本估计：
$$I(\omega_j) = \frac{1}{n}\left|\sum_{t=1}^{n}y_t e^{-i\omega_j t}\right|^2$$

在傅里叶频率 $\omega_j = 2\pi j/n$，其中 $j = 0, 1, \ldots, n/2$。

**关键频率：**
- $\omega = 0$：均值水平（零频率）
- $\omega = 2\pi/m$：周期为 m 个时间单位
- $\omega = \pi$：奈奎斯特频率（最快可观测周期 = 2个时间单位）

## 数学与推导

### 傅里叶变换关系

自协方差和谱密度是傅里叶变换对：
$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$
$$\gamma(h) = \int_{-\pi}^{\pi}f(\omega)e^{i\omega h}d\omega$$

**帕塞瓦尔关系：**
$$\gamma(0) = \text{Var}(y_t) = \int_{-\pi}^{\pi}f(\omega)d\omega$$

总方差在各频率上分解。

### AR(1)的谱密度

对于 $y_t = \phi y_{t-1} + \epsilon_t$：
$$f(\omega) = \frac{\sigma^2}{2\pi|1-\phi e^{-i\omega}|^2} = \frac{\sigma^2}{2\pi(1+\phi^2-2\phi\cos\omega)}$$

性质：
- $\phi > 0$：峰值在 $\omega = 0$（低频主导）
- $\phi < 0$：峰值在 $\omega = \pi$（高频主导）

### MA(1)的谱密度

对于 $y_t = \epsilon_t + \theta\epsilon_{t-1}$：
$$f(\omega) = \frac{\sigma^2}{2\pi}|1+\theta e^{-i\omega}|^2 = \frac{\sigma^2}{2\pi}(1+\theta^2+2\theta\cos\omega)$$

### 周期检测

如果周期图在 $\omega_j$ 处有峰值，主要周期为：
$$T = \frac{2\pi}{\omega_j} = \frac{n}{j}$$

对于具有年度周期的月度数据：峰值在 $\omega = 2\pi/12 \approx 0.524$。

## 算法/模型框架

**谱分析步骤：**

```
1. 去除均值（必要时去除趋势）
   y_centered = y - mean(y)

2. 应用窗函数（可选，减少泄漏）
   常用：Hanning, Hamming, Blackman

3. 计算FFT
   Y = FFT(y_centered)

4. 计算周期图
   I[j] = |Y[j]|² / n

5. 平滑周期图（可选）
   - Daniell核
   - 对数平滑
   - Welch方法

6. 识别峰值
   - 与红/白噪声基准比较
   - 检验显著性（对连续谱进行F检验）

7. 解释
   - 峰值 → 主要周期
   - 平滑衰减 → AR类行为
   - 平坦 → 白噪声
```

**频率到周期的转换：**
$$\text{周期} = \frac{n}{\text{索引}} = \frac{2\pi}{\omega}$$

## 常见陷阱

1. **谱泄漏**：由于有限样本，真实谱中的尖锐峰值会扩散。使用窗函数可以减少泄漏。

2. **混淆周期图与谱密度**：周期图是不一致的（不收敛）。需要平滑处理以估计密度。

3. **忽略混叠**：比奈奎斯特频率更快的周期（周期 < 2）会出现在错误的频率上。确保充分采样。

4. **非平稳性**：谱分析假设平稳性。趋势会导致低频膨胀。需要先去趋势。

5. **过度解读峰值**：随机波动会产生虚假峰值。需要对噪声基准进行显著性检验。

6. **错误的频率解释**：$\omega = 0.5$ 不意味着周期 = 0.5。周期 = $2\pi/0.5 \approx 12.6$。

## 简例

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 生成具有已知频率的信号
np.random.seed(42)
n = 500
t = np.arange(n)

# 成分：趋势 + 周期50 + 周期12 + 噪声
y = (0.01 * t +                          # 趋势
     5 * np.sin(2 * np.pi * t / 50) +    # 周期50
     3 * np.sin(2 * np.pi * t / 12) +    # 周期12
     np.random.randn(n))                  # 噪声

# 去趋势
y_detrended = signal.detrend(y)

# 计算周期图
freqs, psd = signal.periodogram(y_detrended, fs=1.0)

# 查找峰值
peaks, _ = signal.find_peaks(psd, height=np.percentile(psd, 90))

print("检测到的周期：")
for p in peaks:
    if freqs[p] > 0:
        period = 1 / freqs[p]
        print(f"  频率 {freqs[p]:.4f} → 周期 {period:.1f}")

# 预期：在周期50和周期12附近有峰值
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 自协方差函数和谱密度之间有什么关系？</summary>

<div class="answer">
<strong>答案：</strong> 它们是傅里叶变换对。

$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$

$$\gamma(h) = \int_{-\pi}^{\pi}f(\omega)e^{i\omega h}d\omega$$

**解释：**
- ACF描述时域中的相关性
- 谱密度描述频域中的功率
- 相同信息，不同表示

**关键洞见：** 当 $h=0$ 时：
$$\gamma(0) = \text{Var}(y_t) = \int f(\omega)d\omega$$

总方差 = 谱密度的积分（所有频率上的功率）。

<div class="pitfall">
<strong>常见误区：</strong> 认为频域和时域分析给出不同的信息。它们是等价的表示——根据问题选择更易解释的方式。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 为什么正φ的AR(1)在零频率处有峰值？</summary>

<div class="answer">
<strong>答案：</strong> 正的AR(1)系数产生持续性——数值倾向于在均值上方或下方停留较长时间。这对应于慢振荡（低频）。

**数学解释：**
谱密度：$f(\omega) \propto \frac{1}{1+\phi^2-2\phi\cos\omega}$

在 $\omega = 0$ 时：$f(0) \propto \frac{1}{(1-\phi)^2}$（对 $\phi > 0$ 最大）
在 $\omega = \pi$ 时：$f(\pi) \propto \frac{1}{(1+\phi)^2}$（对 $\phi > 0$ 最小）

**直觉：**
- $\phi > 0$：今天的值预测明天的值 → 平滑、低频行为
- $\phi < 0$：今天的值预测明天相反 → 波动、高频行为

<div class="pitfall">
<strong>常见误区：</strong> 期望所有AR过程的谱形状相似。φ的符号和大小会显著改变谱形状。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 推导奈奎斯特频率并解释为什么无法检测高于它的频率。</summary>

<div class="answer">
<strong>答案：</strong> 奈奎斯特频率 = $\pi$ 弧度/样本 = 0.5 周期/样本。

**推导：**
要观测一个周期，我们每个周期至少需要2个样本（在峰值和谷值处采样）。

最小可检测周期 = 2个采样间隔
最大频率 = 1/(2个采样间隔) = 0.5 周期/样本

用角频率表示：$\omega_{奈奎斯特} = 2\pi \times 0.5 = \pi$

**混叠：**
频率 $\omega > \pi$ 的信号会显示为频率 $2\pi - \omega$（被反射）。

例如：真实频率0.6周期/样本会显示为0.4周期/样本。

**结论：** 没有更高的采样率，我们无法区分 $\omega$ 和 $2\pi - \omega$。

<div class="pitfall">
<strong>常见误区：</strong> 试图从月度数据中检测日周期。月度采样（奈奎斯特 = 2个月周期）无法看到比双月振荡更快的任何东西。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 为什么原始周期图是谱密度的不一致估计量？</summary>

<div class="answer">
<strong>答案：</strong> 周期图的方差不会随着样本量的增加而减少。

**技术解释：**
$$\text{Var}(I(\omega)) \approx f(\omega)^2$$

对于 $\omega \neq 0, \pi$。方差等于均值的平方——相对误差保持恒定！

**为什么会这样：**
- 每个频率的周期图使用整个序列的信息
- 但在每个傅里叶频率，我们本质上只有一个"观测"
- 更多数据 → 更多频率，但每个频率仍然只有一个估计

**解决方案：** 平滑周期图
- 平均邻近频率（减少方差）
- 或使用多窗方法
- 以偏差换取方差

<div class="pitfall">
<strong>常见误区：</strong> 将原始周期图峰值视为确定性结果。峰值可能是噪声；始终进行平滑或显著性检验。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你分析每小时温度数据，看到周期24小时处有强峰值，但也有周期12、8、6小时处的意外峰值。发生了什么？</summary>

<div class="answer">
<strong>答案：</strong> 这些是基本日周期的**谐波**。

**解释：**
纯24小时周期只会在周期24处显示一个峰值。但真实的温度模式不是纯正弦波——它们有：
- 早晨急剧上升
- 下午缓慢下降
- 这些非正弦形状需要多个频率来表示

**傅里叶定理：** 任何周期信号都是谐波的和：
$$y(t) = \sum_{k=1}^{\infty} a_k \cos(2\pi kt/24) + b_k \sin(2\pi kt/24)$$

谐波在周期 24/2=12, 24/3=8, 24/4=6 等处。

**解释：**
- 周期24：基本日周期
- 周期12：不对称性（早晨≠傍晚）
- 周期8、6：更多形状细节

**行动：** 这对于非正弦周期是正常的。关注基本频率；谐波表示形状。

<div class="pitfall">
<strong>常见误区：</strong> 将谐波解释为单独的物理现象。它们是非正弦形状的数学产物。
</div>
</div>
</details>

## 参考文献

1. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. 第4章.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. 第4章.
3. Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.
4. Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical Applications*. Cambridge University Press.
