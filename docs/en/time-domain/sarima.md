# Seasonal ARIMA (SARIMA) Models

<div class="interview-summary">
<strong>Interview Summary:</strong> SARIMA(p,d,q)(P,D,Q)[s] adds seasonal AR and MA terms at lag s (e.g., s=12 for monthly). Seasonal differencing $(1-L^s)$ removes seasonal unit roots. The "airline model" SARIMA(0,1,1)(0,1,1)[12] is a benchmark for seasonal data. Identification uses ACF/PACF at both regular and seasonal lags.
</div>

## Core Definitions

**SARIMA(p,d,q)(P,D,Q)[s] Model**:
$$\Phi(L)\Phi_s(L^s)(1-L)^d(1-L^s)^D X_t = c + \Theta(L)\Theta_s(L^s)\epsilon_t$$

**Components:**
- $(p, d, q)$: Non-seasonal AR order, differencing, MA order
- $(P, D, Q)$: Seasonal AR order, differencing, MA order
- $s$: Seasonal period (e.g., 12 for monthly, 4 for quarterly)

**Polynomials:**
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ (non-seasonal AR)
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$ (non-seasonal MA)
- $\Phi_s(L^s) = 1 - \Phi_1 L^s - \cdots - \Phi_P L^{Ps}$ (seasonal AR)
- $\Theta_s(L^s) = 1 + \Theta_1 L^s + \cdots + \Theta_Q L^{Qs}$ (seasonal MA)

**Seasonal Difference Operator**:
$$\nabla_s X_t = (1 - L^s)X_t = X_t - X_{t-s}$$

## Math and Derivations

### SARIMA(0,0,0)(1,0,0)[12]: Seasonal AR(1)

$$X_t = \Phi_1 X_{t-12} + \epsilon_t$$

ACF significant only at lags 12, 24, 36, ... with exponential decay.

### SARIMA(0,0,0)(0,0,1)[12]: Seasonal MA(1)

$$X_t = \epsilon_t + \Theta_1\epsilon_{t-12}$$

ACF significant only at lag 12, zero elsewhere.

### SARIMA(0,1,1)(0,1,1)[12]: The Airline Model

The classic Box-Jenkins airline passenger model:
$$(1-L)(1-L^{12})X_t = (1+\theta L)(1+\Theta L^{12})\epsilon_t$$

Expanding:
$$X_t - X_{t-1} - X_{t-12} + X_{t-13} = \epsilon_t + \theta\epsilon_{t-1} + \Theta\epsilon_{t-12} + \theta\Theta\epsilon_{t-13}$$

**Key properties:**
- First difference handles trend
- Seasonal difference handles annual pattern
- MA(1) smooths non-seasonal noise
- Seasonal MA(1) smooths annual noise
- Cross-term $\theta\Theta$ at lag 13

### ACF/PACF Patterns for SARIMA

**Pure seasonal AR (0,0,0)(P,0,0)[s]:**
- ACF: Exponential decay at seasonal lags (s, 2s, 3s, ...)
- PACF: Cuts off at lag Ps

**Pure seasonal MA (0,0,0)(0,0,Q)[s]:**
- ACF: Cuts off at lag Qs
- PACF: Exponential decay at seasonal lags

**Mixed SARIMA:**
- Non-seasonal patterns at lags 1, 2, 3, ...
- Seasonal patterns at lags s, 2s, 3s, ...
- Interaction patterns at lags s±1, s±2, ...

### Multiplicative Model Structure

The multiplicative form means:
$$\Phi(L)\Phi_s(L^s) = (1-\phi_1 L)(1 - \Phi_1 L^s) = 1 - \phi_1 L - \Phi_1 L^s + \phi_1\Phi_1 L^{s+1}$$

This creates interaction terms (e.g., coefficient at lag 13 for monthly data with AR(1) × SAR(1)).

## Algorithm/Model Sketch

**Identification Procedure:**

```
1. Plot series; identify seasonal period s
2. Check for trend → apply regular differencing (d)
3. Check for seasonal pattern → apply seasonal differencing (D)
4. Usually D ≤ 1, d ≤ 2

5. Examine ACF/PACF of stationary series:
   - At lags 1, 2, ..., s-1: determine p, q
   - At lags s, 2s, 3s: determine P, Q
   - Spikes at s±k: interaction effects

6. Fit candidate models
7. Compare AIC/BIC
8. Check residuals at both regular and seasonal lags
```

**Common Seasonal Periods:**

| Data Frequency | Period s |
|----------------|----------|
| Monthly | 12 |
| Quarterly | 4 |
| Weekly (annual) | 52 |
| Daily (weekly) | 7 |
| Hourly (daily) | 24 |

## Common Pitfalls

1. **Double seasonal patterns**: Some data has multiple seasonalities (daily + weekly). Standard SARIMA handles one period. Consider multiple seasonal models or alternative methods.

2. **Large s causes issues**: For s=52 or s=365, estimation is difficult. Consider Fourier terms or alternative decomposition methods.

3. **Seasonal differencing with D > 1**: Rarely needed and often causes over-differencing. Check if D=1 suffices.

4. **Ignoring multiplicative structure**: The model is multiplicative, so lag s+1 effects exist when both $\phi$ and $\Phi$ are non-zero.

5. **Non-integer periods**: If seasonality isn't at integer lags (e.g., 365.25 days/year), SARIMA doesn't apply directly. Use trigonometric seasonality.

6. **Forgetting trend after seasonal differencing**: Seasonal differencing $(1-L^{12})$ doesn't remove linear trend. May still need $d=1$.

## Mini Example

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate SARIMA(1,1,1)(1,1,1)[12] data
np.random.seed(42)
n = 200
s = 12

# Create seasonal + trend + noise
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / s)
trend = 0.1 * t
noise = np.random.randn(n) * 2
X = trend + seasonal + np.cumsum(noise)

# Fit SARIMA
model = SARIMAX(X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
print(results.summary().tables[1])

# Check residuals at seasonal lags
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(results.resid, lags=[12, 24], return_df=True)
print("\nLjung-Box test at seasonal lags:")
print(lb_test)

# Forecast
forecast = results.get_forecast(steps=12)
print(f"\n12-month forecast mean: {forecast.predicted_mean.values}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why is the SARIMA model called "multiplicative"? How does this affect the lag structure?</summary>

<div class="answer">
<strong>Answer:</strong> "Multiplicative" refers to multiplying AR and MA polynomials: $\Phi(L) \times \Phi_s(L^s)$. This creates interaction terms at combined lags.

<strong>Example:</strong> SARIMA(1,0,0)(1,0,0)[12]:
$$(1-\phi L)(1-\Phi L^{12})X_t = \epsilon_t$$
$$X_t - \phi X_{t-1} - \Phi X_{t-12} + \phi\Phi X_{t-13} = \epsilon_t$$

The $\phi\Phi$ term at lag 13 is an interaction effect—it wouldn't exist in an additive model.

**Implication:** Check ACF/PACF at lags like 11, 13 (not just 12) for monthly data.

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting clean separation of seasonal and non-seasonal effects. The multiplicative structure blends them, which can confuse identification.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What is the "airline model" and why is it a useful benchmark?</summary>

<div class="answer">
<strong>Answer:</strong> The airline model is SARIMA(0,1,1)(0,1,1)[12], originally fit to airline passenger data by Box and Jenkins. It's useful because:

1. Handles trend via first differencing
2. Handles annual seasonality via seasonal differencing
3. Uses only 2 parameters ($\theta$, $\Theta$) yet fits many seasonal series well
4. Equivalent to Holt-Winters exponential smoothing

**Model:**
$$(1-L)(1-L^{12})X_t = (1+\theta L)(1+\Theta L^{12})\epsilon_t$$

<div class="pitfall">
<strong>Common pitfall:</strong> Using the airline model as the default without checking fit. For some data, AR terms or different differencing may be needed. Always verify with residual diagnostics.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the ACF at lag 12 for the seasonal MA(1) model: $X_t = \epsilon_t + \Theta\epsilon_{t-12}$.</summary>

<div class="answer">
<strong>Answer:</strong> $\rho(12) = \frac{\Theta}{1+\Theta^2}$, and $\rho(h) = 0$ for $h \neq 0, 12$.

<strong>Derivation:</strong>

**Variance:**
$$\gamma(0) = \text{Var}(\epsilon_t + \Theta\epsilon_{t-12}) = \sigma^2(1 + \Theta^2)$$

**Autocovariance at lag 12:**
$$\gamma(12) = E[(\epsilon_t + \Theta\epsilon_{t-12})(\epsilon_{t-12} + \Theta\epsilon_{t-24})]$$
$$= E[\Theta\epsilon_{t-12}^2] = \Theta\sigma^2$$

**ACF:**
$$\rho(12) = \frac{\gamma(12)}{\gamma(0)} = \frac{\Theta\sigma^2}{\sigma^2(1+\Theta^2)} = \frac{\Theta}{1+\Theta^2}$$

For other lags, there's no overlap of $\epsilon$ terms, so $\gamma(h) = 0$.

<div class="pitfall">
<strong>Common pitfall:</strong> Note $|\rho(12)| \leq 0.5$, same constraint as non-seasonal MA(1). Larger observed seasonal correlations suggest seasonal AR or combined model.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> For SARIMA(0,1,0)(0,1,0)[12], write out the model equation and explain what it represents.</summary>

<div class="answer">
<strong>Answer:</strong> This is a "seasonal random walk":
$$(1-L)(1-L^{12})X_t = \epsilon_t$$

Expanding:
$$X_t - X_{t-1} - X_{t-12} + X_{t-13} = \epsilon_t$$
$$X_t = X_{t-1} + X_{t-12} - X_{t-13} + \epsilon_t$$

**Interpretation:** Today's value = yesterday's value + this month last year − same day last year + noise.

This is the "naïve seasonal" forecast: $\hat{X}_{t+1} = X_t + (X_{t+1-12} - X_{t-12})$.

It says: repeat last year's seasonal pattern while continuing yesterday's level.

<div class="pitfall">
<strong>Common pitfall:</strong> This model is a useful benchmark but often too simplistic. Real data usually benefits from MA terms to smooth noise.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You're modeling hourly electricity demand with clear daily (24h) and weekly (168h) patterns. Can you use standard SARIMA? What alternatives exist?</summary>

<div class="answer">
<strong>Answer:</strong> Standard SARIMA handles only one seasonal period. For multiple seasonalities, alternatives include:

1. **Double seasonal models**: SARIMA with s=24 plus external regressors for weekly pattern
2. **TBATS**: Exponential smoothing with multiple seasonal periods
3. **Fourier terms**: Include sin/cos terms at both frequencies
4. **Prophet**: Handles multiple seasonalities via additive decomposition
5. **Neural approaches**: LSTM or Transformers can learn complex patterns

**Practical approach:**
- Use s=24 (dominant pattern)
- Add day-of-week dummies or Fourier terms for weekly pattern
- Consider: `SARIMAX(p,d,q)(P,D,Q)[24]` with `exog=weekly_dummies`

<div class="pitfall">
<strong>Common pitfall:</strong> Trying to use s=168 for weekly seasonality causes estimation problems (168 is large). Use hierarchical or additive approaches instead.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 9.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 9.
3. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. Chapter 3.
4. De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *JASA*, 106(496), 1513-1527.
