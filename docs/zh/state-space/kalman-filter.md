# 状态空间模型与卡尔曼滤波

<div class="interview-summary">
<strong>面试要点：</strong> 状态空间模型通过随时间演变的隐藏状态来表示时间序列。卡尔曼滤波递归估计状态：预测步骤（向前传播状态），更新步骤（纳入新观测）。对于线性高斯系统是最优的。核心方程：$x_t = Fx_{t-1} + w_t$，$y_t = Hx_t + v_t$。统一了ARIMA、指数平滑和结构模型。
</div>

## 核心定义

**状态空间表示：**

状态方程：$\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t$，$\mathbf{w}_t \sim N(0, \mathbf{Q})$

观测方程：$y_t = \mathbf{H}\mathbf{x}_t + v_t$，$v_t \sim N(0, R)$

**组成部分：**
- $\mathbf{x}_t$：状态向量（未观测）
- $y_t$：观测值（数据）
- $\mathbf{F}$：状态转移矩阵
- $\mathbf{H}$：观测矩阵
- $\mathbf{Q}$：状态噪声协方差
- $R$：观测噪声方差

**局部水平模型（最简单）：**
$$\mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)$$
$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2_\epsilon)$$

## 数学与推导

### 卡尔曼滤波递推

**预测步骤：**
$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}' + \mathbf{Q}$$

**更新步骤：**
$$\mathbf{K}_t = \mathbf{P}_{t|t-1}\mathbf{H}'(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}' + R)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(y_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t\mathbf{H})\mathbf{P}_{t|t-1}$$

**关键量：**
- $\hat{\mathbf{x}}_{t|t}$：滤波状态估计（使用截至t的数据）
- $\mathbf{P}_{t|t}$：状态协方差（不确定性）
- $\mathbf{K}_t$：卡尔曼增益（新观测的权重）
- $v_t = y_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1}$：新息（预测误差）

### 局部水平模型的卡尔曼滤波

状态：$\mathbf{x}_t = \mu_t$（标量）
矩阵：$F = 1$，$H = 1$，$Q = \sigma^2_\eta$，$R = \sigma^2_\epsilon$

递推：
$$\hat{\mu}_{t|t-1} = \hat{\mu}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + \sigma^2_\eta$$
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma^2_\epsilon}$$
$$\hat{\mu}_{t|t} = \hat{\mu}_{t|t-1} + K_t(y_t - \hat{\mu}_{t|t-1})$$
$$P_{t|t} = (1 - K_t)P_{t|t-1}$$

当 $t \to \infty$：$K_t \to K^* = $ 稳态增益（与SES的α相关）。

### 局部线性趋势模型

状态：$\mathbf{x}_t = (\mu_t, \beta_t)'$（水平和趋势）

$$\begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} \mu_{t-1} \\ \beta_{t-1} \end{pmatrix} + \begin{pmatrix} \eta_t \\ \zeta_t \end{pmatrix}$$

$$y_t = (1, 0)\begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix} + \epsilon_t$$

## 算法/模型框架

**卡尔曼滤波算法：**

```
输入：y[1:n], F, H, Q, R, x0, P0
输出：滤波状态，新息，似然

初始化：
  x_hat = x0
  P = P0
  log_lik = 0

对于 t = 1 到 n：
  # 预测
  x_pred = F @ x_hat
  P_pred = F @ P @ F' + Q

  # 新息
  v = y[t] - H @ x_pred
  S = H @ P_pred @ H' + R

  # 更新
  K = P_pred @ H' @ inv(S)
  x_hat = x_pred + K @ v
  P = (I - K @ H) @ P_pred

  # 似然贡献
  log_lik += -0.5 * (log(det(S)) + v' @ inv(S) @ v)

返回 x_hat_history, P_history, log_lik
```

**卡尔曼平滑器**（使用所有数据）：
$$\hat{\mathbf{x}}_{t|n} = \hat{\mathbf{x}}_{t|t} + \mathbf{J}_t(\hat{\mathbf{x}}_{t+1|n} - \hat{\mathbf{x}}_{t+1|t})$$

其中 $\mathbf{J}_t = \mathbf{P}_{t|t}\mathbf{F}'\mathbf{P}_{t+1|t}^{-1}$

## 常见陷阱

1. **数值不稳定性**：协方差矩阵可能变得非正定。使用平方根或UD分解。

2. **错误的初始化**：不良的 $\mathbf{x}_0, \mathbf{P}_0$ 会影响早期估计。对于未知初始状态使用扩散初始化。

3. **模型误设**：卡尔曼滤波仅对真实模型是最优的。非高斯或非线性系统需要扩展（EKF、UKF、粒子滤波）。

4. **忘记平滑器**：对于历史分析（非实时），使用卡尔曼平滑器来纳入未来观测。

5. **过于复杂的状态空间**：可以表示许多模型，但简单的替代方案（ARIMA）可能更容易。

6. **混淆滤波与预测**：滤波 = 给定截至t的数据的估计。预测 = 超出观测数据的预测。

## 简例

```python
import numpy as np

def kalman_filter_local_level(y, sigma_eta, sigma_eps, mu0=None, P0=1000):
    """局部水平模型的卡尔曼滤波。"""
    n = len(y)

    # 初始化
    mu_filt = np.zeros(n)
    P_filt = np.zeros(n)
    mu_pred = mu0 if mu0 is not None else y[0]
    P_pred = P0

    for t in range(n):
        # 预测（当 t > 0 时）
        if t > 0:
            mu_pred = mu_filt[t-1]
            P_pred = P_filt[t-1] + sigma_eta**2

        # 更新
        K = P_pred / (P_pred + sigma_eps**2)
        mu_filt[t] = mu_pred + K * (y[t] - mu_pred)
        P_filt[t] = (1 - K) * P_pred

    return mu_filt, P_filt

# 生成局部水平数据
np.random.seed(42)
n = 100
sigma_eta, sigma_eps = 0.5, 1.0

# 真实状态
mu_true = np.cumsum(np.random.randn(n) * sigma_eta)
# 观测值
y = mu_true + np.random.randn(n) * sigma_eps

# 运行卡尔曼滤波
mu_hat, P_hat = kalman_filter_local_level(y, sigma_eta, sigma_eps)

print("卡尔曼滤波结果：")
print(f"最终状态估计：{mu_hat[-1]:.2f}")
print(f"真实最终状态：{mu_true[-1]:.2f}")
print(f"最终状态标准差：{np.sqrt(P_hat[-1]):.3f}")

# 稳态卡尔曼增益
K_steady = (-sigma_eps**2 + np.sqrt(sigma_eps**4 + 4*sigma_eta**2*sigma_eps**2)) / (2*sigma_eps**2)
print(f"\n稳态卡尔曼增益：{K_steady:.3f}")
print(f"等效SES的alpha：{K_steady:.3f}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 卡尔曼增益背后的直觉是什么？</summary>

<div class="answer">
<strong>答案：</strong> 卡尔曼增益 $K_t$ 平衡对预测的信任和对观测的信任。

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R} = \frac{\text{预测不确定性}}{\text{预测不确定性} + \text{观测噪声}}$$

**解释：**
- 高 $K_t$（接近1）：更信任观测；状态显著更新
- 低 $K_t$（接近0）：更信任预测；观测影响很小

**何时K高？**
- 状态不确定性高（$P$ 大）
- 观测噪声低（$R$ 小）

**何时K低？**
- 状态不确定性低（$P$ 小）
- 观测噪声高（$R$ 大）

<div class="pitfall">
<strong>常见误区：</strong> 认为增益是固定的。它随不确定性变化而演变。早期，增益可能较高（先验不确定）；后来，随着滤波器"学习"而稳定。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 局部水平模型与简单指数平滑有什么关系？</summary>

<div class="answer">
<strong>答案：</strong> 它们是等价的！具有特定噪声比的局部水平模型产生SES。

**局部水平：**
$$\mu_t = \mu_{t-1} + \eta_t, \quad y_t = \mu_t + \epsilon_t$$

**卡尔曼更新：**
$$\hat{\mu}_{t|t} = \hat{\mu}_{t|t-1} + K(y_t - \hat{\mu}_{t|t-1})$$

**在稳态时：** $K \to K^* = \alpha$（SES平滑参数）

关系：
$$\alpha = \frac{-\sigma_\epsilon^2 + \sqrt{\sigma_\epsilon^4 + 4\sigma_\eta^2\sigma_\epsilon^2}}{2\sigma_\eta^2}$$

**关键洞见：** SES是局部水平模型的稳态卡尔曼滤波。卡尔曼提供：
- 最优初始化
- 适当的不确定性量化
- 与似然的联系

<div class="pitfall">
<strong>常见误区：</strong> 使用SES而不认识它假设特定的状态空间结构。如果动态不同，SES可能不是最优的。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 推导局部水平模型的稳态卡尔曼增益。</summary>

<div class="answer">
<strong>答案：</strong> 在稳态时，$P_{t|t} = P^*$（常数）。

**递推：**
$$P_{t|t-1} = P_{t-1|t-1} + Q = P^* + Q$$
$$P_{t|t} = (1 - K_t)P_{t|t-1} = P^*$$

**稳态条件：**
$$P^* = (1 - K^*)(P^* + Q)$$
$$P^* = P^* + Q - K^*(P^* + Q)$$
$$K^* = \frac{Q}{P^* + Q}$$

另外：$K^* = \frac{P^* + Q}{P^* + Q + R}$

求解 $P^*$：
$$P^* = \frac{-R + \sqrt{R^2 + 4QR}}{2}$$

以及：
$$K^* = \frac{-R + \sqrt{R^2 + 4QR}}{2Q}$$

对于 Q = $\sigma_\eta^2$，R = $\sigma_\epsilon^2$，这给出SES的α关系。

<div class="pitfall">
<strong>常见误区：</strong> 稳态可能需要多次迭代才能达到。对于短序列，时变增益很重要。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 滤波状态估计和平滑状态估计有什么区别？</summary>

<div class="answer">
<strong>答案：</strong>

**滤波：** $\hat{x}_{t|t} = E[x_t | y_1, \ldots, y_t]$
- 使用截至时间t的观测
- 实时估计
- 来自前向卡尔曼递推

**平滑：** $\hat{x}_{t|n} = E[x_t | y_1, \ldots, y_n]$
- 使用所有观测
- 回顾性估计
- 前向递推后需要后向递推

**关键区别：** 平滑器使用未来观测来改进过去的状态估计。

**何时使用哪个：**
- 实时预测：滤波
- 历史分析：平滑
- 参数估计：平滑（更好的似然）

**方差关系：**
$$\text{Var}(\hat{x}_{t|n}) \leq \text{Var}(\hat{x}_{t|t})$$

平滑器永远不会比滤波器差。

<div class="pitfall">
<strong>常见误区：</strong> 当平滑器可用且更准确时，仍使用滤波估计进行历史分析。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你有已知精度(R)的GPS测量，但车辆动态(Q)未知。你将如何调整卡尔曼滤波器？</summary>

<div class="answer">
<strong>答案：</strong>

**方法1：最大似然**
- 将Q视为参数
- 从新息计算似然：$\log L = -\frac{1}{2}\sum(\log S_t + v_t^2/S_t)$
- 优化Q以最大化似然

**方法2：基于新息的调参**
- 新息 $v_t$ 应该是方差为 $S_t$ 的白噪声
- 如果新息自相关：Q太小
- 如果新息方差 >> $S_t$：Q太小
- 如果新息方差 << $S_t$：Q太大

**方法3：自适应滤波**
- 从最近的新息统计量在线估计Q
- $\hat{Q}_t = $ 最近新息的样本方差减去R

**实践步骤：**
1. 从 Q = R 开始（相等信任）
2. 检查新息统计量
3. 网格搜索或梯度优化
4. 在保留数据上验证

<div class="pitfall">
<strong>常见误区：</strong> Q设置太小会使滤波器过度自信且跟踪变化缓慢。太大会使其噪声。交叉验证！
</div>
</div>
</details>

## 参考文献

1. Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
2. Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
3. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
4. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. 第13章.
