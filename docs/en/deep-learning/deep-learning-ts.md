# Deep Learning for Time Series

<div class="interview-summary">
<strong>Interview Summary:</strong> Deep learning models (RNN, LSTM, TCN, Transformer) capture complex temporal patterns. LSTM addresses vanishing gradients via gates. TCN uses dilated causal convolutions for long-range dependencies. Transformers use attention mechanisms. DL shines with: large data, multiple series, complex patterns. May underperform classical methods on small data or simple patterns.
</div>

## Core Definitions

**RNN (Recurrent Neural Network):**
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$\hat{y}_t = W_y h_t$$

**LSTM (Long Short-Term Memory):**
- Forget gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- Input gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- Cell update: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- Cell state: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- Output gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- Hidden state: $h_t = o_t * \tanh(C_t)$

**TCN (Temporal Convolutional Network):**
Dilated causal convolutions with residual connections.

**Transformer:** Self-attention mechanism for sequence modeling.

## Math and Derivations

### Vanishing Gradient Problem in RNN

Gradient of loss with respect to early hidden state:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T}\prod_{k=t}^{T-1}\frac{\partial h_{k+1}}{\partial h_k}$$

For tanh activation: $|\frac{\partial h_{k+1}}{\partial h_k}| < 1$ typically

Product of many small numbers → gradient vanishes.

**LSTM solution:** Cell state path has additive updates (not multiplicative), preserving gradients over long sequences.

### TCN Dilated Convolutions

For dilation factor $d$ and filter size $k$:
$$(F *_d x)_t = \sum_{i=0}^{k-1} f_i \cdot x_{t-d \cdot i}$$

**Receptive field with L layers:**
$$R = 1 + (k-1)\sum_{l=0}^{L-1}d_l$$

With exponential dilation ($d_l = 2^l$): $R = 1 + (k-1)(2^L - 1)$

### Transformer Self-Attention

**Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For time series:
- Q, K, V derived from input sequence
- Attention weights show which past timesteps are relevant
- Position encoding added to maintain temporal order

### Training Strategies

**Teacher forcing:** During training, use actual values (not predictions) as input.

**Multi-step loss:** Optimize over multiple forecast horizons:
$$L = \sum_{h=1}^{H}w_h \cdot L_h(\hat{y}_{t+h}, y_{t+h})$$

## Algorithm/Model Sketch

**LSTM for Forecasting:**

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
        # Take last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out

# Training loop
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

**When to Use Deep Learning:**

| Scenario | Recommendation |
|----------|---------------|
| Small data (<1000) | Classical (ARIMA, ETS) |
| Medium data, simple patterns | Classical or simple NN |
| Large data, complex patterns | Deep learning |
| Many related series | Deep learning (transfer) |
| Real-time, low latency | TCN (parallelizable) |

## Common Pitfalls

1. **Too little data:** DL needs thousands+ of observations. With small data, ARIMA often wins.

2. **Over-complicated architecture:** Simple LSTM often beats complex Transformer on univariate forecasting.

3. **Ignoring baselines:** Always compare to naive, seasonal naive, and ARIMA before claiming DL success.

4. **Lookback window too short:** LSTM can learn long patterns only if lookback is long enough.

5. **Not using validation properly:** Use time-aware validation (rolling origin), not random split.

6. **Training instability:** Gradient clipping, learning rate scheduling, and careful initialization matter.

## Mini Example

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Generate data
np.random.seed(42)
n = 2000
t = np.arange(n)
y = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 10) + np.random.randn(n) * 0.3

# Create sequences
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

# Train-test split
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Simple LSTM model
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

# Train
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
for epoch in range(20):
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), Y_batch)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_rmse = torch.sqrt(criterion(y_pred, Y_test))
    print(f"Test RMSE: {test_rmse:.4f}")

# Compare to naive
naive_rmse = np.sqrt(np.mean((Y_test.numpy() - X_test[:, -1, :].numpy())**2))
print(f"Naive RMSE: {naive_rmse:.4f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the vanishing gradient problem and how does LSTM address it?</summary>

<div class="answer">
<strong>Answer:</strong>

**Problem:** In vanilla RNN, gradients are multiplied through each timestep. With many steps, gradients become exponentially small (vanish), preventing learning of long-range dependencies.

**LSTM Solution:**
- Cell state $C_t$ is updated additively, not multiplicatively
- Forget gate controls what to keep: $C_t = f_t * C_{t-1} + ...$
- When $f_t \approx 1$, gradient flows unchanged
- Information can persist over hundreds of timesteps

**Key insight:** The cell state acts as a "highway" for gradients, bypassing the vanishing problem.

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking LSTM completely solves long-range dependencies. Very long sequences (1000+) may still need attention mechanisms or hierarchical structures.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> When would you choose TCN over LSTM for time series?</summary>

<div class="answer">
<strong>Answer:</strong>

**Prefer TCN when:**
1. **Parallelization needed:** TCN processes all timesteps simultaneously; LSTM is sequential
2. **Long sequences:** Dilated convolutions efficiently handle very long range
3. **Training stability:** TCN gradients don't explode/vanish as easily
4. **Inference speed:** No hidden state to maintain
5. **Variable length at inference:** Can process any length

**Prefer LSTM when:**
1. **Truly sequential processing:** Online/streaming data
2. **Variable length training:** LSTM naturally handles different lengths
3. **State tracking needed:** Hidden state captures "memory"
4. **Smaller receptive field sufficient:** LSTM may use parameters more efficiently

**Research finding:** TCN often matches or beats LSTM on standard benchmarks with faster training.

<div class="pitfall">
<strong>Common pitfall:</strong> Defaulting to LSTM because it's "the standard." TCN is often simpler and faster with comparable accuracy.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> How does dilated convolution increase receptive field without increasing parameters?</summary>

<div class="answer">
<strong>Answer:</strong> Dilation introduces gaps in the convolution, sampling inputs at regular intervals.

**Standard convolution (dilation=1):**
$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-i}$$
Receptive field = k

**Dilated convolution (dilation=d):**
$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-d \cdot i}$$
Receptive field = 1 + (k-1) × d

**With exponential dilation (d = 2^l):**
- Layer 0: RF = k
- Layer 1: RF = k + (k-1)×2
- Layer L-1: RF = 1 + (k-1)(2^L - 1)

**Example:** k=3, L=8 → RF = 1 + 2×255 = 511

Same number of parameters (k weights per layer), but 500+ timestep receptive field!

<div class="pitfall">
<strong>Common pitfall:</strong> Using too many layers. With k=3 and L=10, RF ≈ 2000. Check if you actually need that range.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Explain why Transformers need positional encoding for time series.</summary>

<div class="answer">
<strong>Answer:</strong> Self-attention is permutation invariant—it treats input as a set, not a sequence.

**Problem:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This doesn't change if we permute the input order—attention weights depend only on pairwise similarities.

**Solution: Positional encoding**
Add position information to embeddings:
$$x'_t = x_t + PE(t)$$

Common encoding:
$$PE(t, 2i) = \sin(t / 10000^{2i/d})$$
$$PE(t, 2i+1) = \cos(t / 10000^{2i/d})$$

Now the model can distinguish $x_5$ from $x_{50}$ even if content is identical.

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting positional encoding → Transformer treats sequence as bag of vectors, losing temporal structure entirely.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You train an LSTM on energy demand data and it predicts flat lines (always the mean). What's wrong?</summary>

<div class="answer">
<strong>Answer:</strong> Several possible causes:

1. **Learning rate too high:** Weights oscillate, model defaults to mean
   - Fix: Reduce LR, use scheduler

2. **Vanishing gradients:** Despite LSTM, can still occur
   - Fix: Gradient clipping, check gradient norms

3. **Data not scaled:** Large values cause saturation
   - Fix: Standardize inputs and targets

4. **Lookback too short:** Model can't see useful patterns
   - Fix: Increase sequence length

5. **Too few epochs:** Model hasn't learned yet
   - Fix: Train longer, check loss curve

6. **Wrong loss function:** MSE on non-stationary data dominated by trend
   - Fix: Use differenced data or relative errors

7. **Model too small:** Can't capture complexity
   - Fix: Increase hidden size / layers

**Diagnosis:**
- Plot training loss: decreasing or flat?
- Check gradient magnitudes
- Visualize predictions vs. actuals over training

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming mean prediction is "failure." For high-noise data with no pattern, mean IS optimal. Compare to naive baselines first.
</div>
</div>
</details>

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions A*, 379(2194).
