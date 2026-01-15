# Simple Exponential Smoothing (SES)

<div class="interview-summary">
<strong>Interview Summary:</strong> SES forecasts using weighted average of past observations with exponentially decaying weights. Formula: $\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$. Parameter $\alpha \in (0,1)$ controls responsiveness. Equivalent to ARIMA(0,1,1). Best for series with no trend or seasonality. Point forecasts are flat (constant) for all horizons.
</div>

## Core Definitions

**Simple Exponential Smoothing**:
$$\hat{y}_{t+1|t} = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}$$

**Alternative Forms:**

Component form:
$$\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$$
$$\hat{y}_{t+h|t} = \ell_t$$

Weighted average form:
$$\hat{y}_{t+1|t} = \alpha \sum_{j=0}^{t-1}(1-\alpha)^j y_{t-j} + (1-\alpha)^t \ell_0$$

**Parameters:**
- $\alpha \in (0,1)$: Smoothing parameter
- $\ell_0$: Initial level

## Math and Derivations

### Exponential Weights

Expanding the recursion:
$$\hat{y}_{t+1|t} = \alpha y_t + \alpha(1-\alpha)y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots$$

Weights: $\alpha, \alpha(1-\alpha), \alpha(1-\alpha)^2, \ldots$

These sum to 1: $\alpha \sum_{j=0}^{\infty}(1-\alpha)^j = \alpha \cdot \frac{1}{1-(1-\alpha)} = 1$

### Connection to ARIMA(0,1,1)

ARIMA(0,1,1): $(1-L)y_t = (1+\theta L)\epsilon_t$

The optimal forecast is:
$$\hat{y}_{t+1|t} = \hat{y}_{t|t-1} + \frac{1}{1+\theta}(y_t - \hat{y}_{t|t-1})$$

With $\alpha = \frac{1}{1+\theta}$, this is identical to SES.

For invertibility ($|\theta| < 1$): $\alpha \in (0.5, 1)$

### Forecast Error Variance

For ARIMA(0,1,1):
$$\text{Var}(y_{t+h} - \hat{y}_{t+h|t}) = \sigma^2[1 + (h-1)(1-\alpha)^2]$$

Prediction interval:
$$\hat{y}_{t+h|t} \pm z_{\alpha/2}\sigma\sqrt{1 + (h-1)(1-\alpha)^2}$$

### Optimal Smoothing Parameter

Choose $\alpha$ to minimize sum of squared one-step forecast errors:
$$\text{SSE} = \sum_{t=1}^{n}(y_t - \hat{y}_{t|t-1})^2$$

No closed form; use numerical optimization.

## Algorithm/Model Sketch

**SES Algorithm:**

```
Input: time series y[1:n], smoothing parameter α
Output: forecasts

1. Initialize: ℓ[0] = y[1] (or average of first few values)

2. For t = 1 to n:
   ℓ[t] = α * y[t] + (1-α) * ℓ[t-1]
   fitted[t] = ℓ[t-1]  # one-step-ahead

3. For h = 1 to H:
   forecast[n+h] = ℓ[n]  # flat forecast

Return forecasts
```

**Selecting α:**
- $\alpha \to 0$: Heavy smoothing, slow response to changes
- $\alpha \to 1$: Light smoothing, forecasts close to most recent observation
- Typical range: 0.1 to 0.3

## Common Pitfalls

1. **Using SES with trend**: SES produces flat forecasts. For trending data, use Holt's method.

2. **Using SES with seasonality**: SES doesn't capture seasonal patterns. Use Holt-Winters or seasonal decomposition first.

3. **Choosing α arbitrarily**: Always optimize α using historical data or use cross-validation.

4. **Ignoring initialization**: The choice of $\ell_0$ affects early forecasts. Common choices: $\ell_0 = y_1$ or $\ell_0 = \bar{y}$.

5. **Expecting decreasing forecast intervals**: For SES, prediction intervals grow with horizon (like random walk).

6. **Confusing α interpretation**: High α = less smoothing (more weight on recent data). Some practitioners expect the opposite.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Generate level + noise data (appropriate for SES)
np.random.seed(42)
n = 100
level = 50
y = level + np.random.randn(n) * 5

# Fit SES with optimization
model = SimpleExpSmoothing(y, initialization_method='estimated')
fit = model.fit(optimized=True)

print(f"Optimal alpha: {fit.params['smoothing_level']:.3f}")
print(f"Initial level: {fit.params['initial_level']:.2f}")

# Compare different alpha values
alphas = [0.1, 0.3, 0.5, 0.9]
for alpha in alphas:
    fit_alpha = model.fit(smoothing_level=alpha, optimized=False)
    sse = np.sum((y - fit_alpha.fittedvalues)**2)
    print(f"Alpha={alpha}: SSE={sse:.1f}")

# Forecast
forecast = fit.forecast(10)
print(f"\n10-step forecast (all same): {forecast.values}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why are SES forecasts constant (flat) for all future horizons?</summary>

<div class="answer">
<strong>Answer:</strong> SES models the series as a local level plus noise: $y_t = \ell_t + \epsilon_t$. The best estimate of future level is the current level $\ell_T$. With no trend or seasonality modeled, there's no reason to predict change.

<strong>Explanation:</strong>
The forecast equation:
$$\hat{y}_{T+h|T} = \ell_T \text{ for all } h \geq 1$$

This assumes the level stays constant. The uncertainty (prediction interval) grows with h, but the point forecast doesn't change.

<div class="pitfall">
<strong>Common pitfall:</strong> Using flat forecasts for trending data leads to systematic under/over-prediction. Always check if the series has a trend before choosing SES.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Explain the trade-off when choosing the smoothing parameter α.</summary>

<div class="answer">
<strong>Answer:</strong>
- **High α (close to 1)**: More weight on recent observations. Fast response to changes but noisy forecasts. Good for series with frequent level shifts.
- **Low α (close to 0)**: More weight on distant observations. Smooth forecasts but slow to adapt. Good for stable series with much noise.

**The trade-off:**
- Responsiveness vs. stability
- Bias vs. variance (high α = high variance; low α = potential bias if level changes)

**Optimal α:** Balances these considerations; found by minimizing forecast error on historical data.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming lower α is always "smoother" and better. In volatile series, low α causes the forecast to lag behind actual level shifts.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Show that the weights in SES sum to 1.</summary>

<div class="answer">
<strong>Answer:</strong> The forecast is $\hat{y} = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j y_{t-j}$.

<strong>Derivation:</strong>
Sum of weights:
$$\sum_{j=0}^{\infty}\alpha(1-\alpha)^j = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j$$

This is a geometric series with ratio $(1-\alpha)$, where $|1-\alpha| < 1$:
$$= \alpha \cdot \frac{1}{1-(1-\alpha)} = \alpha \cdot \frac{1}{\alpha} = 1$$

**Key equation:** $\sum_{j=0}^{\infty}r^j = \frac{1}{1-r}$ for $|r| < 1$.

<div class="pitfall">
<strong>Common pitfall:</strong> In practice, we don't have infinite history. The "missing" weight goes to the initialization $\ell_0$, which is why initialization matters for short series.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Derive the relationship between SES parameter α and ARIMA(0,1,1) parameter θ.</summary>

<div class="answer">
<strong>Answer:</strong> $\alpha = \frac{1}{1+\theta}$ or equivalently $\theta = \frac{1-\alpha}{\alpha}$.

<strong>Derivation:</strong>
ARIMA(0,1,1): $y_t - y_{t-1} = \epsilon_t + \theta\epsilon_{t-1}$

The optimal h-step forecast:
$$\hat{y}_{t+1|t} = y_t + \theta\hat{\epsilon}_t$$

where $\hat{\epsilon}_t = y_t - \hat{y}_{t|t-1}$

This gives:
$$\hat{y}_{t+1|t} = y_t + \theta(y_t - \hat{y}_{t|t-1}) = (1+\theta)y_t - \theta\hat{y}_{t|t-1}$$

Rearranging:
$$\hat{y}_{t+1|t} = \frac{1}{1+\theta}(1+\theta)y_t + \frac{\theta}{1+\theta}\hat{y}_{t|t-1}$$

With $\alpha = \frac{1}{1+\theta}$ and $1-\alpha = \frac{\theta}{1+\theta}$, this matches SES.

<div class="pitfall">
<strong>Common pitfall:</strong> ARIMA(0,1,1) with $\theta > 0$ gives $\alpha < 0.5$, which is non-invertible. Standard SES with optimized α usually gives $\alpha > 0.5$ (invertible range).
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You apply SES to monthly sales data and get optimal α = 0.95. What does this suggest about your data? What should you consider?</summary>

<div class="answer">
<strong>Answer:</strong> α = 0.95 means almost all weight on the most recent observation. This suggests:

1. **High volatility or frequent level shifts** in the data
2. **Possible trend** that SES is trying to track by being very responsive
3. **Potential outliers** pulling the optimization toward high α
4. **Near-random-walk behavior**

**What to consider:**
1. Check for trend → use Holt's method instead
2. Look for outliers → they inflate optimal α
3. Plot the series and fitted values → see if SES is "chasing" the data
4. Try Holt's or Holt-Winters → may give better forecasts with lower α
5. Consider ARIMA(0,1,1) → the θ would be near 0, confirming random walk

<div class="pitfall">
<strong>Common pitfall:</strong> Very high α often indicates model misspecification (missing trend or seasonality), not that SES is appropriate. Investigate before accepting.
</div>
</div>
</details>

## References

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 8.
2. Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing*. Springer.
3. Gardner, E. S. (1985). Exponential smoothing: The state of the art. *Journal of Forecasting*, 4(1), 1-28.
4. Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill.
