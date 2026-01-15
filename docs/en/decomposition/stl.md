# STL Decomposition

<div class="interview-summary">
<strong>Interview Summary:</strong> STL (Seasonal and Trend decomposition using Loess) robustly separates a series into trend, seasonal, and remainder components using local regression. Unlike classical decomposition, STL handles any seasonal period and is robust to outliers. Key parameters: seasonal window (odd integer ≥ 7) and trend window. Remainder should be stationary for forecasting.
</div>

## Core Definitions

**Additive Decomposition:**
$$y_t = T_t + S_t + R_t$$

- $T_t$: Trend component (smooth, long-term movement)
- $S_t$: Seasonal component (periodic pattern, sums to ~0 over each cycle)
- $R_t$: Remainder/residual (everything else)

**Loess (Locally Estimated Scatterplot Smoothing):**
Local polynomial regression using weighted least squares. Weights decrease with distance from target point.

**Key STL Parameters:**
- `seasonal`: Seasonal period (e.g., 12 for monthly)
- `seasonal_deg`: Degree of seasonal polynomial (0 or 1)
- `trend`: Trend smoothing window (odd integer, default depends on seasonal)
- `robust`: Whether to use robust fitting (downweight outliers)

## Math and Derivations

### Loess Smoother

For point $x_0$, fit weighted polynomial:
$$\min_{\beta} \sum_{i=1}^{n} w_i(x_0)(y_i - \beta_0 - \beta_1(x_i - x_0))^2$$

Weights use tricube function:
$$w(u) = \begin{cases} (1-|u|^3)^3 & |u| < 1 \\ 0 & |u| \geq 1 \end{cases}$$

Distance scaled by bandwidth $h$: $u_i = |x_i - x_0|/h$

### STL Algorithm (Simplified)

**Outer loop** (for robustness):
1. Initialize: $R_t^{(0)} = 0$, $T_t^{(0)} = $ loess smooth of $y_t$

**Inner loop**:
2. **Detrend**: $y_t - T_t^{(k-1)}$
3. **Cycle-subseries smoothing**: For each season $s=1,\ldots,m$, smooth values at positions $s, s+m, s+2m, \ldots$ using loess
4. **Low-pass filter**: Remove low-frequency from seasonal
5. **Deseasonalize**: $y_t - S_t^{(k)}$
6. **Trend extraction**: Loess smooth of deseasonalized series

Repeat inner loop until convergence; outer loop updates robustness weights.

### Robustness Weights

After each outer iteration, compute residuals and assign weights:
$$\rho_t = |R_t|$$
$$h = 6 \cdot \text{median}(|\rho_t|)$$
$$w_t = B(\rho_t/h)$$

where $B$ is the bisquare function: $B(u) = (1-u^2)^2$ for $|u| < 1$, else 0.

Outliers get downweighted in subsequent iterations.

## Algorithm/Model Sketch

**STL Decomposition Steps:**

```
Input: y[1:n], seasonal period m, parameters
Output: Trend T, Seasonal S, Remainder R

1. Initialize trend T = loess(y) or moving average
2. For k = 1 to n_outer:

   For j = 1 to n_inner:
      a. Detrend: D = y - T
      b. For each position i in 1...m:
         - Extract subseries: values at i, i+m, i+2m,...
         - Smooth subseries with loess
         - Store smoothed seasonal values
      c. Low-pass filter seasonal (remove trend leakage)
      d. Subtract filtered seasonal from raw seasonal → S
      e. Deseasonalize: y - S
      f. Smooth deseasonalized → T

   Update robustness weights based on R = y - T - S

3. Return T, S, R = y - T - S
```

## Common Pitfalls

1. **Wrong seasonal period**: STL requires correct m. If m is wrong, seasonal won't be captured properly.

2. **Over-smoothing trend**: Too large trend window removes real variation. Under-smoothing captures noise.

3. **Seasonal leakage into trend**: If seasonal window too small, trend absorbs some seasonality.

4. **Not using robust mode**: Outliers distort both trend and seasonal. Always try `robust=True` first.

5. **Assuming multiplicative works directly**: STL is additive. For multiplicative, log-transform first, then decompose, then exponentiate.

6. **Ignoring remainder**: Remainder should look like noise. Strong patterns indicate model inadequacy.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Generate data with trend, seasonality, and outliers
np.random.seed(42)
n = 120
t = np.arange(n)
trend = 50 + 0.3 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(n) * 3
y = trend + seasonal + noise

# Add outliers
y[50] += 40
y[80] -= 35

# STL decomposition (robust)
stl = STL(y, period=12, robust=True)
result = stl.fit()

print("Component statistics:")
print(f"Trend range: [{result.trend.min():.1f}, {result.trend.max():.1f}]")
print(f"Seasonal range: [{result.seasonal.min():.1f}, {result.seasonal.max():.1f}]")
print(f"Remainder std: {result.resid.std():.2f}")

# Check if outliers are in remainder (they should be)
print(f"\nRemainder at outlier positions:")
print(f"  t=50: {result.resid[50]:.1f}")
print(f"  t=80: {result.resid[80]:.1f}")

# Plot
fig = result.plot()
plt.tight_layout()
plt.show()
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why is STL preferred over classical decomposition in many applications?</summary>

<div class="answer">
<strong>Answer:</strong> STL advantages:

1. **Flexibility**: Works with any seasonal period, not just 4 or 12
2. **Robustness**: Outliers don't distort estimates (with robust=True)
3. **Control**: Adjustable smoothness via window parameters
4. **Evolving seasonal**: Can capture slowly changing seasonal patterns
5. **No end-point issues**: Loess handles boundaries better than moving averages

Classical decomposition:
- Assumes fixed seasonal pattern
- Sensitive to outliers
- Moving average loses observations at ends
- Limited to standard frequencies

<div class="pitfall">
<strong>Common pitfall:</strong> Using classical decomposition by default. STL is almost always better, especially with outliers or evolving patterns.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> How does the robustness mechanism in STL work?</summary>

<div class="answer">
<strong>Answer:</strong> Iterative reweighting:

1. First pass: fit STL normally, compute residuals
2. Identify outliers: large |residuals| relative to median
3. Assign weights: outliers get weight → 0, normal points → 1
4. Refit STL with weighted observations
5. Repeat until convergence

**Weight function (bisquare):**
$$w = (1 - (r/h)^2)^2$$

where $r$ = |residual|, $h$ = 6 × median(|residuals|)

Outliers (large $r$) get near-zero weights and don't influence the fit.

<div class="pitfall">
<strong>Common pitfall:</strong> Not using robust mode then wondering why one outlier distorts the entire seasonal pattern. Always start with robust=True.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> In loess smoothing, why is the tricube weight function used?</summary>

<div class="answer">
<strong>Answer:</strong> Tricube function $w(u) = (1-|u|^3)^3$:

1. **Smooth**: Continuous and differentiable, giving smooth fitted curves
2. **Compact support**: Zero beyond bandwidth, so distant points don't influence fit
3. **Downweighting**: Smoothly decreases influence with distance
4. **Computationally nice**: Simple polynomial form

**Properties:**
- $w(0) = 1$ (full weight at target point)
- $w(u) \to 0$ smoothly as $|u| \to 1$
- $w(u) = 0$ for $|u| \geq 1$

Alternative: Gaussian weights have infinite support (all points contribute), which is less local.

<div class="pitfall">
<strong>Common pitfall:</strong> Choosing bandwidth too small → jagged fit; too large → over-smoothed. STL uses data-driven defaults but tuning may help.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why does STL use a low-pass filter on the seasonal component?</summary>

<div class="answer">
<strong>Answer:</strong> The low-pass filter removes trend that leaked into the seasonal during cycle-subseries smoothing.

**Problem:** When smoothing each season's subseries (e.g., all Januaries), if there's trend, the subseries average drifts. This drift appears as low-frequency content in the seasonal.

**Solution:** Apply moving average to the seasonal across full cycles:
$$L_t = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} S^*_{t+j}$$

Then subtract: $S_t = S^*_t - L_t$

This ensures seasonal averages to zero over each cycle.

<div class="pitfall">
<strong>Common pitfall:</strong> Without the low-pass filter, seasonal component captures some trend, leaving residual with trend pattern.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> After STL decomposition, your remainder shows a clear AR(1) pattern. What does this mean and what should you do?</summary>

<div class="answer">
<strong>Answer:</strong> An AR(1) pattern in remainder means:
- STL captured trend and seasonality
- But short-term autocorrelation remains
- This is common and expected

**What to do:**
1. **For forecasting**: Model remainder with AR(1) or ARIMA
   - Forecast trend (extrapolation or drift)
   - Forecast seasonal (repeat pattern)
   - Forecast remainder with AR(1)
   - Combine: $\hat{y} = \hat{T} + \hat{S} + \hat{R}$

2. **STL + ARIMA pipeline:**
   ```python
   stl_result = STL(y, period=12).fit()
   remainder = stl_result.resid
   arima_model = ARIMA(remainder, order=(1,0,0)).fit()
   ```

3. **Consider ETS/SARIMA directly**: They handle all components in one model.

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring remainder autocorrelation in forecasting. This underestimates short-term uncertainty and biases predictions.
</div>
</div>
</details>

## References

1. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
2. Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: An approach to regression analysis by local fitting. *JASA*, 83(403), 596-610.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 3.
4. Dokumentov, A., & Hyndman, R. J. (2015). STR: A seasonal-trend decomposition procedure based on regression. *Monash Econometrics Working Papers*.
