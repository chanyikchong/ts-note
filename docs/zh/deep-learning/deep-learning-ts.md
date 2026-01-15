# 深度学习与时间序列

<div class="interview-summary">
<strong>面试要点：</strong> 深度学习模型（RNN、LSTM、TCN、Transformer）能够捕获复杂的时间模式。LSTM通过门控机制解决梯度消失问题。TCN使用膨胀因果卷积处理长程依赖。Transformer使用注意力机制。深度学习在以下情况表现出色：大数据量、多序列、复杂模式。在小数据或简单模式上可能不如传统方法。
</div>

## 核心定义

**RNN（循环神经网络）：**
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$\hat{y}_t = W_y h_t$$

**LSTM（长短期记忆网络）：**
- 遗忘门：$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- 输入门：$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- 单元更新：$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- 单元状态：$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- 输出门：$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- 隐藏状态：$h_t = o_t * \tanh(C_t)$

**TCN（时间卷积网络）：**
带残差连接的膨胀因果卷积。

**Transformer：** 用于序列建模的自注意力机制。

## 数学与推导

### RNN中的梯度消失问题

损失对早期隐藏状态的梯度：
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T}\prod_{k=t}^{T-1}\frac{\partial h_{k+1}}{\partial h_k}$$

对于tanh激活：$|\frac{\partial h_{k+1}}{\partial h_k}| < 1$ 通常成立

多个小数相乘 → 梯度消失。

**LSTM的解决方案：** 单元状态路径采用加性更新（非乘性），在长序列上保持梯度。

### TCN膨胀卷积

对于膨胀因子 $d$ 和滤波器大小 $k$：
$$(F *_d x)_t = \sum_{i=0}^{k-1} f_i \cdot x_{t-d \cdot i}$$

**L层的感受野：**
$$R = 1 + (k-1)\sum_{l=0}^{L-1}d_l$$

采用指数膨胀（$d_l = 2^l$）：$R = 1 + (k-1)(2^L - 1)$

### Transformer自注意力

**注意力：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于时间序列：
- Q、K、V从输入序列派生
- 注意力权重显示哪些过去的时间步是相关的
- 添加位置编码以保持时间顺序

### 训练策略

**教师强制：** 训练时使用实际值（而非预测值）作为输入。

**多步损失：** 在多个预测时间范围上优化：
$$L = \sum_{h=1}^{H}w_h \cdot L_h(\hat{y}_{t+h}, y_{t+h})$$

## 算法/模型概述

**LSTM预测：**

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个隐藏状态
        out = self.fc(lstm_out[:, -1, :])
        return out

# 训练循环
model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
```

**何时使用深度学习：**

| 场景 | 建议 |
|----------|---------------|
| 小数据（<1000） | 传统方法（ARIMA、ETS） |
| 中等数据，简单模式 | 传统方法或简单神经网络 |
| 大数据，复杂模式 | 深度学习 |
| 多个相关序列 | 深度学习（迁移学习） |
| 实时，低延迟 | TCN（可并行化） |

## 常见陷阱

1. **数据太少：** 深度学习需要数千以上的观测值。小数据时ARIMA通常胜出。

2. **架构过于复杂：** 简单LSTM在单变量预测上通常优于复杂Transformer。

3. **忽略基线：** 在声称深度学习成功之前，始终与朴素预测、季节性朴素和ARIMA比较。

4. **回望窗口太短：** LSTM只有在回望足够长时才能学习长模式。

5. **验证方法不当：** 使用时间感知验证（滚动原点），而不是随机分割。

6. **训练不稳定：** 梯度裁剪、学习率调度和仔细的初始化很重要。

## 小型示例

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 生成数据
np.random.seed(42)
n = 2000
t = np.arange(n)
y = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 10) + np.random.randn(n) * 0.3

# 创建序列
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

seq_length = 50
X, Y = create_sequences(y, seq_length)
X = torch.FloatTensor(X).unsqueeze(-1)  # (N, seq_len, 1)
Y = torch.FloatTensor(Y).unsqueeze(-1)  # (N, 1)

# 训练-测试分割
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# 简单LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = SimpleLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 训练
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
for epoch in range(20):
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), Y_batch)
        loss.backward()
        optimizer.step()

# 评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_rmse = torch.sqrt(criterion(y_pred, Y_test))
    print(f"测试RMSE: {test_rmse:.4f}")

# 与朴素预测比较
naive_rmse = np.sqrt(np.mean((Y_test.numpy() - X_test[:, -1, :].numpy())**2))
print(f"朴素RMSE: {naive_rmse:.4f}")
```

## 测验

<details class="quiz">
<summary><strong>问题1（概念题）：</strong> 什么是梯度消失问题，LSTM如何解决它？</summary>

<div class="answer">
<strong>答案：</strong>

**问题：** 在普通RNN中，梯度在每个时间步相乘。步数很多时，梯度变得指数级小（消失），阻止学习长程依赖。

**LSTM的解决方案：**
- 单元状态 $C_t$ 采用加性更新，而非乘性更新
- 遗忘门控制保留什么：$C_t = f_t * C_{t-1} + ...$
- 当 $f_t \approx 1$ 时，梯度不变地流动
- 信息可以在数百个时间步上持续

**核心洞察：** 单元状态充当梯度的"高速公路"，绕过消失问题。

<div class="pitfall">
<strong>常见陷阱：</strong> 认为LSTM完全解决了长程依赖问题。非常长的序列（1000+）可能仍需要注意力机制或分层结构。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题2（概念题）：</strong> 什么时候会选择TCN而不是LSTM用于时间序列？</summary>

<div class="answer">
<strong>答案：</strong>

**优先选择TCN当：**
1. **需要并行化：** TCN同时处理所有时间步；LSTM是顺序的
2. **长序列：** 膨胀卷积高效处理非常长的范围
3. **训练稳定性：** TCN的梯度不容易爆炸/消失
4. **推理速度：** 无需维护隐藏状态
5. **推理时变长：** 可以处理任意长度

**优先选择LSTM当：**
1. **真正的顺序处理：** 在线/流式数据
2. **训练时变长：** LSTM自然处理不同长度
3. **需要状态跟踪：** 隐藏状态捕获"记忆"
4. **较小感受野足够：** LSTM可能更高效地使用参数

**研究发现：** TCN在标准基准上通常与LSTM相当或更好，且训练更快。

<div class="pitfall">
<strong>常见陷阱：</strong> 默认选择LSTM因为它是"标准"。TCN通常更简单、更快，精度相当。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题3（数学题）：</strong> 膨胀卷积如何在不增加参数的情况下扩大感受野？</summary>

<div class="answer">
<strong>答案：</strong> 膨胀在卷积中引入间隙，以规则间隔采样输入。

**标准卷积（膨胀=1）：**
$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-i}$$
感受野 = k

**膨胀卷积（膨胀=d）：**
$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-d \cdot i}$$
感受野 = 1 + (k-1) × d

**指数膨胀（d = 2^l）：**
- 第0层：RF = k
- 第1层：RF = k + (k-1)×2
- 第L-1层：RF = 1 + (k-1)(2^L - 1)

**示例：** k=3, L=8 → RF = 1 + 2×255 = 511

相同数量的参数（每层k个权重），但500+时间步的感受野！

<div class="pitfall">
<strong>常见陷阱：</strong> 使用太多层。k=3和L=10时，RF ≈ 2000。检查是否真的需要那么大的范围。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题4（数学题）：</strong> 解释为什么Transformer在时间序列中需要位置编码。</summary>

<div class="answer">
<strong>答案：</strong> 自注意力是置换不变的——它将输入视为集合，而非序列。

**问题：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

如果我们置换输入顺序，这不会改变——注意力权重仅取决于成对相似性。

**解决方案：位置编码**
向嵌入添加位置信息：
$$x'_t = x_t + PE(t)$$

常用编码：
$$PE(t, 2i) = \sin(t / 10000^{2i/d})$$
$$PE(t, 2i+1) = \cos(t / 10000^{2i/d})$$

现在模型可以区分 $x_5$ 和 $x_{50}$，即使内容相同。

<div class="pitfall">
<strong>常见陷阱：</strong> 忘记位置编码 → Transformer将序列视为向量袋，完全丢失时间结构。
</div>
</div>
</details>

<details class="quiz">
<summary><strong>问题5（实践题）：</strong> 你在能源需求数据上训练LSTM，它预测出平线（总是均值）。出了什么问题？</summary>

<div class="answer">
<strong>答案：</strong> 几种可能的原因：

1. **学习率太高：** 权重振荡，模型默认为均值
   - 解决：降低学习率，使用调度器

2. **梯度消失：** 尽管是LSTM，仍可能发生
   - 解决：梯度裁剪，检查梯度范数

3. **数据未缩放：** 大值导致饱和
   - 解决：标准化输入和目标

4. **回望太短：** 模型看不到有用的模式
   - 解决：增加序列长度

5. **训练轮次太少：** 模型还没学到
   - 解决：训练更久，检查损失曲线

6. **损失函数不当：** 非平稳数据上的MSE被趋势主导
   - 解决：使用差分数据或相对误差

7. **模型太小：** 无法捕获复杂性
   - 解决：增加隐藏层大小/层数

**诊断：**
- 绘制训练损失：下降还是持平？
- 检查梯度大小
- 可视化训练过程中的预测vs实际

<div class="pitfall">
<strong>常见陷阱：</strong> 假设均值预测是"失败"。对于高噪声无模式的数据，均值是最优的。首先与朴素基线比较。
</div>
</div>
</details>

## 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions A*, 379(2194).
