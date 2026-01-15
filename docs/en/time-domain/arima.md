# ARIMA Models

<div class="interview-summary">
<strong>Interview Summary:</strong> ARIMA(p,d,q) extends ARMA to non-stationary series by including differencing. The "I" stands for "integrated" — meaning the series needs d differences to become stationary. After differencing d times, fit ARMA(p,q) to the differenced series. Most commonly d=1 (first difference) or d=2 (second difference).
</div>

## Core Definitions

**ARIMA(p,d,q) Model**:

Apply ARMA(p,q) to the d-th difference of $X_t$:
$$\Phi(L)(1-L)^d X_t = c + \Theta(L)\epsilon_t$$

**Components:**
- $p$: AR order (autoregressive lags)
- $d$: Degree of differencing (integration order)
- $q$: MA order (moving average lags)

**Difference Operator**:
$$\nabla X_t = (1-L)X_t = X_t - X_{t-1}$$
$$\nabla^2 X_t = X_t - 2X_{t-1} + X_{t-2}$$

**Integrated Process**: A process is integrated of order d, written $I(d)$, if it requires d differences to become stationary.

## Math and Derivations

### ARIMA(0,1,0): Random Walk

$$X_t = X_{t-1} + \epsilon_t$$

Or equivalently: $(1-L)X_t = \epsilon_t$

The first difference $\nabla X_t = \epsilon_t$ is white noise (stationary).

### ARIMA(0,1,0) with Drift

$$X_t = c + X_{t-1} + \epsilon_t$$

The drift $c$ creates a linear trend in levels:
$$E[X_t] = X_0 + ct$$

### ARIMA(1,1,0): Differenced AR(1)

$$\nabla X_t = \phi \nabla X_{t-1} + \epsilon_t$$

Expanding:
$$(X_t - X_{t-1}) = \phi(X_{t-1} - X_{t-2}) + \epsilon_t$$
$$X_t = (1+\phi)X_{t-1} - \phi X_{t-2} + \epsilon_t$$

This is AR(2) in levels with a unit root.

### ARIMA(0,1,1): IMA(1,1)

$$\nabla X_t = \epsilon_t + \theta\epsilon_{t-1}$$

Also known as an **exponentially weighted moving average (EWMA)** process. Forms the basis for simple exponential smoothing.

### General ARIMA(p,d,q)

In lag operator notation:
$$\Phi(L)(1-L)^d X_t = c + \Theta(L)\epsilon_t$$

Where:
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ has roots outside unit circle (stationary AR)
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$ has roots outside unit circle (invertible MA)
- $(1-L)^d$ contributes $d$ unit roots

### Forecasting with ARIMA

For ARIMA(p,1,q), the h-step forecast:
$$\hat{X}_{T+h|T} = E[X_{T+h} | X_T, X_{T-1}, \ldots]$$

Key property: forecasts revert to a linear trend (if drift) or constant growth for $d \geq 1$.

**Prediction intervals** widen with horizon due to accumulated uncertainty.

## Algorithm/Model Sketch

**Box-Jenkins Methodology:**

```
1. IDENTIFICATION
   - Plot series; check for trend/non-stationarity
   - Apply ADF/KPSS tests
   - Difference until stationary (determine d)
   - Examine ACF/PACF of differenced series
   - Identify candidate (p, q) orders

2. ESTIMATION
   - Fit candidate ARIMA models
   - Use MLE (or CSS for initial values)
   - Check parameter significance

3. DIAGNOSTICS
   - Examine residuals: ACF should show no pattern
   - Ljung-Box test for residual autocorrelation
   - Check residual normality (Q-Q plot)
   - Look for outliers

4. FORECASTING
   - Generate point forecasts
   - Compute prediction intervals
   - Validate on holdout data if possible
```

**Determining d:**

| Symptom | Likely d |
|---------|----------|
| Series wanders, slow ACF decay | d = 1 |
| Trend in differenced series | d = 2 |
| Seasonal pattern persists | Need seasonal differencing |
| Already fluctuates around mean | d = 0 |

## Common Pitfalls

1. **Over-differencing**: If the original series is stationary, differencing introduces MA(1) with $\theta = -1$. Check: if ACF of differenced series has large negative spike at lag 1, you may have over-differenced.

2. **Under-differencing**: ACF that doesn't decay or stays significant at high lags suggests more differencing needed. Also check KPSS test.

3. **Ignoring drift**: ARIMA(0,1,0) without constant is pure random walk. With drift, there's a trend. Misspecifying this affects long-term forecasts.

4. **d > 2 rarely needed**: If you need d > 2, reconsider—series might have other issues (outliers, structural breaks, wrong transformation).

5. **Confusing trend types**: Deterministic trend (detrend with regression) vs. stochastic trend (difference). Using wrong approach gives poor results.

6. **Negative forecasts**: For positive series (prices, counts), ARIMA may forecast negatives. Consider log transform or constrained models.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Generate ARIMA(1,1,1) process
np.random.seed(42)
n = 300
phi, theta = 0.5, 0.3
eps = np.random.randn(n + 2)

# First generate the differenced series as ARMA(1,1)
dX = np.zeros(n)
dX[0] = eps[1] + theta * eps[0]
for t in range(1, n):
    dX[t] = phi * dX[t-1] + eps[t+1] + theta * eps[t]

# Integrate to get X
X = np.cumsum(dX)

# Test stationarity
adf_X = adfuller(X)
adf_dX = adfuller(np.diff(X))
print(f"ADF p-value (levels): {adf_X[1]:.4f}")  # Should be high (non-stationary)
print(f"ADF p-value (differenced): {adf_dX[1]:.4f}")  # Should be low (stationary)

# Fit ARIMA(1,1,1)
model = ARIMA(X, order=(1, 1, 1)).fit()
print(f"\nTrue: phi={phi}, theta={theta}")
print(f"Estimated: phi={model.arparams[0]:.3f}, theta={model.maparams[0]:.3f}")

# Forecast
forecast = model.forecast(steps=10)
conf_int = model.get_forecast(10).conf_int()
print(f"\n10-step forecast: {forecast[-1]:.2f}")
print(f"95% CI: [{conf_int.iloc[-1, 0]:.2f}, {conf_int.iloc[-1, 1]:.2f}]")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What does the "I" in ARIMA stand for, and what does it mean for a process to be "integrated of order d"?</summary>

<div class="answer">
<strong>Answer:</strong> "I" stands for "Integrated." A process is integrated of order d, denoted I(d), if it requires exactly d differences to become stationary. Integration is the inverse of differencing — if you sum (integrate) a stationary series, you get an I(1) process.

<strong>Explanation:</strong>
- I(0): Stationary (no differencing needed)
- I(1): First difference is stationary (e.g., random walk)
- I(2): Second difference is stationary (e.g., random walk with drift in levels)

**Key insight:** "Integrated" comes from continuous-time analogy. In discrete time: $X_t = \sum_{s=1}^t \epsilon_s$ (integrated/summed white noise) is I(1).

<div class="pitfall">
<strong>Common pitfall:</strong> Confusing integration order with polynomial degree. I(1) is not about linear trends—it's about the type of non-stationarity (stochastic vs. deterministic).
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> How can you tell if a series has been over-differenced?</summary>

<div class="answer">
<strong>Answer:</strong> Signs of over-differencing:
1. ACF of differenced series shows large negative spike at lag 1 (often near -0.5)
2. Variance increases after differencing (should decrease or stay similar)
3. The differenced series looks "over-corrected" with excessive alternation

<strong>Explanation:</strong>
Differencing a stationary series adds MA(1) structure with $\theta \approx -1$:
$$(1-L)X_t = X_t - X_{t-1}$$

If $X_t$ was already stationary, the difference behaves like $\epsilon_t - \epsilon_{t-1}$, which is MA(1) with $\theta = -1$ and $\rho(1) = -0.5$.

**Test:** If $d=1$ differencing gives ACF(1) ≈ -0.5 and all other ACF ≈ 0, try $d=0$.

<div class="pitfall">
<strong>Common pitfall:</strong> Automatically differencing because "everyone does it." Always test stationarity first. Many series (especially returns) are already stationary.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Show that ARIMA(0,1,1) with parameter $\theta$ produces forecasts equivalent to exponential smoothing with $\alpha = 1/(1+\theta)$.</summary>

<div class="answer">
<strong>Answer:</strong> ARIMA(0,1,1): $(1-L)X_t = (1+\theta L)\epsilon_t$

The optimal forecast can be written recursively as:
$$\hat{X}_{t+1|t} = \hat{X}_{t|t-1} + (1+\theta)^{-1}(X_t - \hat{X}_{t|t-1})$$

This is exactly exponential smoothing: $\hat{X}_{t+1} = \alpha X_t + (1-\alpha)\hat{X}_t$ with $\alpha = \frac{1}{1+\theta}$.

<strong>Derivation:</strong>
From ARIMA(0,1,1): $X_t = X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$

The forecast error is:
$$e_t = X_t - \hat{X}_{t|t-1} = \epsilon_t$$

The forecast update:
$$\hat{X}_{t+1|t} = X_t + \theta\hat{\epsilon}_t = X_t + \theta e_t$$

Rearranging:
$$\hat{X}_{t+1|t} = X_t + \theta(X_t - \hat{X}_{t|t-1})/(1+\theta) \cdot (1+\theta)$$

With $\alpha = 1/(1+\theta)$: this gives the exponential smoothing recursion.

<div class="pitfall">
<strong>Common pitfall:</strong> For invertibility, need $|\theta| < 1$, which means $\alpha \in (0.5, 1)$ for IMA(1,1). Values $\alpha < 0.5$ correspond to non-invertible MA.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why do prediction intervals for ARIMA models widen as the forecast horizon increases?</summary>

<div class="answer">
<strong>Answer:</strong> Future shocks $\epsilon_{T+1}, \epsilon_{T+2}, \ldots$ are unknown, and their accumulated effect grows with horizon. For I(d) processes, shocks have permanent effects, causing variance to grow without bound.

<strong>Derivation for random walk (ARIMA(0,1,0)):</strong>
$$X_{T+h} = X_T + \sum_{j=1}^{h}\epsilon_{T+j}$$

Forecast: $\hat{X}_{T+h|T} = X_T$

Error: $X_{T+h} - \hat{X}_{T+h|T} = \sum_{j=1}^{h}\epsilon_{T+j}$

Variance: $\text{Var}(X_{T+h} - \hat{X}_{T+h|T}) = h\sigma^2$

**95% PI:** $X_T \pm 1.96\sigma\sqrt{h}$

The interval width grows like $\sqrt{h}$, becoming arbitrarily wide.

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting narrow long-horizon intervals. ARIMA cannot provide tight long-range forecasts—uncertainty is fundamental. This is why judgment and scenarios matter for long-term planning.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You have monthly sales data showing a clear upward trend. After first differencing, the ACF shows significant spikes at lags 1, 12, and 13. What model structure might be appropriate?</summary>

<div class="answer">
<strong>Answer:</strong> The pattern suggests SARIMA with:
- d=1 (first differencing handles trend)
- Significant lag 1 suggests AR(1) or MA(1)
- Significant lag 12 suggests seasonal component (monthly data, annual pattern)
- Lag 13 = 12+1 is interaction of seasonal and non-seasonal

**Candidate models:**
- SARIMA(1,1,0)(1,0,0)[12]
- SARIMA(0,1,1)(0,1,1)[12] (airline model)
- SARIMA(1,1,1)(1,1,0)[12]

**Next steps:**
1. Apply seasonal differencing and re-check ACF
2. Fit candidates and compare AIC
3. Check residuals for remaining patterns
4. Validate on holdout data

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring the seasonal spike and fitting non-seasonal ARIMA. The lag-12 autocorrelation will persist in residuals, degrading forecasts.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley. Chapters 4-6.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapters 15, 17.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 9.
4. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 5.
