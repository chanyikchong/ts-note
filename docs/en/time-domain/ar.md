# Autoregressive (AR) Models

<div class="interview-summary">
<strong>Interview Summary:</strong> AR(p) models express current value as a linear combination of p past values plus noise. Stationarity requires roots of characteristic polynomial outside unit circle (equivalently, $|\phi| < 1$ for AR(1)). ACF decays exponentially/sinusoidally; PACF cuts off after lag p. Estimate via Yule-Walker, OLS, or MLE.
</div>

## Core Definitions

**AR(p) Model**:
$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

where $\epsilon_t \sim WN(0, \sigma^2)$ (white noise).

**Lag Operator Form**:
$$\Phi(L)X_t = c + \epsilon_t$$

where $\Phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p$ and $LX_t = X_{t-1}$.

**Characteristic Polynomial**:
$$\Phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p$$

**Stationarity Condition**: All roots of $\Phi(z) = 0$ must lie outside the unit circle (|z| > 1).

**Mean of Stationary AR(p)**:
$$\mu = E[X_t] = \frac{c}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}$$

## Math and Derivations

### AR(1) Model: $X_t = c + \phi X_{t-1} + \epsilon_t$

**Stationarity condition**: $|\phi| < 1$

**Mean**: $\mu = \frac{c}{1-\phi}$

**Variance**:
$$\gamma(0) = \text{Var}(X_t) = \frac{\sigma^2}{1-\phi^2}$$

**Autocovariance**:
$$\gamma(h) = \phi^{|h|} \gamma(0) = \frac{\phi^{|h|} \sigma^2}{1-\phi^2}$$

**ACF**: $\rho(h) = \phi^{|h|}$

**PACF**: $\phi_{11} = \phi$, $\phi_{hh} = 0$ for $h > 1$

### AR(2) Model: $X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$

**Stationarity conditions** (all must hold):
1. $\phi_1 + \phi_2 < 1$
2. $\phi_2 - \phi_1 < 1$
3. $|\phi_2| < 1$

**Characteristic roots**: Solutions to $1 - \phi_1 z - \phi_2 z^2 = 0$
- If roots are real: monotonic decay in ACF
- If roots are complex: damped sinusoidal ACF

**Yule-Walker equations for AR(2)**:
$$\rho(1) = \phi_1 + \phi_2\rho(1) \Rightarrow \rho(1) = \frac{\phi_1}{1-\phi_2}$$
$$\rho(2) = \phi_1\rho(1) + \phi_2$$

### General AR(p) Yule-Walker Equations

$$\gamma(h) = \phi_1\gamma(h-1) + \phi_2\gamma(h-2) + \cdots + \phi_p\gamma(h-p) \text{ for } h > 0$$

In matrix form:
$$\begin{pmatrix} \gamma(0) & \gamma(1) & \cdots & \gamma(p-1) \\ \gamma(1) & \gamma(0) & \cdots & \gamma(p-2) \\ \vdots & & \ddots & \vdots \\ \gamma(p-1) & \cdots & \gamma(1) & \gamma(0) \end{pmatrix} \begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{pmatrix} = \begin{pmatrix} \gamma(1) \\ \gamma(2) \\ \vdots \\ \gamma(p) \end{pmatrix}$$

Or in terms of autocorrelations:
$$\mathbf{R}\boldsymbol{\phi} = \boldsymbol{\rho}$$

### Infinite MA Representation

A stationary AR(p) can be written as an infinite MA:
$$X_t = \mu + \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}$$

For AR(1): $\psi_j = \phi^j$

This shows AR processes have infinite memory but with exponentially decaying weights.

## Algorithm/Model Sketch

**Estimation Methods:**

1. **Yule-Walker (Method of Moments)**:
   - Replace $\gamma(h)$ with $\hat{\gamma}(h)$
   - Solve linear system for $\hat{\phi}$
   - Always yields stationary estimates
   - May be inefficient for small samples

2. **Ordinary Least Squares (OLS)**:
   - Regress $X_t$ on $X_{t-1}, \ldots, X_{t-p}$
   - Simple but loses first $p$ observations
   - May give non-stationary estimates

3. **Maximum Likelihood (MLE)**:
   - Most efficient asymptotically
   - Accounts for initial conditions
   - Requires distributional assumption (usually Gaussian)
   - Use numerical optimization

**Order Selection:**
```
1. Examine PACF - significant spikes suggest AR order
2. Fit AR(1), AR(2), ..., AR(p_max)
3. Compare AIC/BIC values
4. Select model with lowest information criterion
5. Verify residuals are white noise
```

## Common Pitfalls

1. **Ignoring stationarity check**: Always verify estimated parameters satisfy stationarity conditions. Non-stationary AR leads to explosive forecasts.

2. **Over-fitting with too many lags**: AIC may favor larger models. BIC penalizes complexity more and often gives better forecasts.

3. **Assuming causality**: AR models capture correlation, not causation. $X_{t-1}$ predicting $X_t$ doesn't mean past causes future.

4. **Neglecting seasonality**: Standard AR doesn't capture seasonal patterns at lag $s$. Consider SARIMA or include $X_{t-s}$ explicitly.

5. **Using OLS without correction**: Standard OLS standard errors are invalid for time series with autocorrelation. Use HAC standard errors or proper likelihood-based inference.

6. **Confusing AR(1) coefficient sign**: Positive $\phi$ gives positive autocorrelation (momentum). Negative $\phi$ gives alternating signs (mean reversion).

## Mini Example

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf

# Generate AR(2) data
np.random.seed(42)
n = 300
phi1, phi2 = 0.6, -0.3
X = np.zeros(n)
eps = np.random.randn(n)

for t in range(2, n):
    X[t] = phi1 * X[t-1] + phi2 * X[t-2] + eps[t]

# Check PACF (should cut off after lag 2)
pacf_values = pacf(X, nlags=10)
print("PACF:", np.round(pacf_values, 3))

# Fit AR model using AIC to select order
from statsmodels.tsa.ar_model import ar_select_order
sel = ar_select_order(X, maxlag=10, ic='aic')
print(f"Selected order: {sel.ar_lags}")

# Fit AR(2) and check estimates
model = AutoReg(X, lags=2).fit()
print(f"True: phi1={phi1}, phi2={phi2}")
print(f"Estimated: phi1={model.params[1]:.3f}, phi2={model.params[2]:.3f}")

# Forecast
forecast = model.forecast(steps=5)
print("5-step forecast:", forecast)
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Explain intuitively why the stationarity condition for AR(1) is $|\phi| < 1$. What happens when $\phi = 1$ or $\phi > 1$?</summary>

<div class="answer">
<strong>Answer:</strong> When $|\phi| < 1$, shocks decay over time, keeping variance bounded. When $\phi = 1$, we have a random walk where shocks persist forever (unit root). When $|\phi| > 1$, the process explodes exponentially.

<strong>Explanation:</strong>
The AR(1) can be written as:
$$X_t = \phi^t X_0 + \sum_{j=0}^{t-1} \phi^j \epsilon_{t-j}$$

- If $|\phi| < 1$: $\phi^t \to 0$ and the MA representation converges (bounded variance)
- If $\phi = 1$: $X_t = X_0 + \sum \epsilon_j$ (random walk, variance $\to \infty$)
- If $|\phi| > 1$: $\phi^t \to \infty$ (explosive)

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking $\phi = 0.99$ is "close enough" to stationary. While technically stationary, near-unit-root processes behave like random walks in finite samples. Predictions degrade quickly.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Why does the PACF of an AR(p) process cut off after lag p while the ACF decays gradually?</summary>

<div class="answer">
<strong>Answer:</strong> PACF measures direct correlation after controlling for intermediate lags. AR(p) by definition only has direct dependence on $p$ past values, so PACF is zero beyond lag $p$. ACF includes indirect effects through intermediate values, causing gradual decay.

<strong>Explanation:</strong>
For AR(2): $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$
- Direct effects: only from $X_{t-1}$ and $X_{t-2}$
- PACF at lag 3: After controlling for $X_{t-1}, X_{t-2}$, $X_{t-3}$ has no additional predictive power
- But ACF at lag 3: $X_t$ correlates with $X_{t-3}$ through the chain $X_{t-1} \to X_{t-2} \to X_{t-3}$

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting perfectly zero PACF beyond lag $p$ in samples. Due to estimation error, you'll see small nonzero values. Use confidence bands to judge significance.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the variance of a stationary AR(1) process: $\text{Var}(X_t) = \frac{\sigma^2}{1-\phi^2}$.</summary>

<div class="answer">
<strong>Answer:</strong> Starting from $X_t = \phi X_{t-1} + \epsilon_t$, take variance of both sides and use stationarity.

<strong>Derivation:</strong>
$$\text{Var}(X_t) = \text{Var}(\phi X_{t-1} + \epsilon_t)$$
$$= \phi^2 \text{Var}(X_{t-1}) + \text{Var}(\epsilon_t) + 2\phi\text{Cov}(X_{t-1}, \epsilon_t)$$

Since $\epsilon_t$ is independent of $X_{t-1}$:
$$\gamma(0) = \phi^2 \gamma(0) + \sigma^2$$

By stationarity, $\text{Var}(X_t) = \text{Var}(X_{t-1}) = \gamma(0)$:
$$\gamma(0) - \phi^2\gamma(0) = \sigma^2$$
$$\gamma(0)(1 - \phi^2) = \sigma^2$$
$$\gamma(0) = \frac{\sigma^2}{1-\phi^2}$$

**Note:** Requires $|\phi| < 1$ for positive variance.

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting that variance increases as $|\phi| \to 1$. Near-unit-root processes have large variance, making them look more volatile than white noise with the same $\sigma^2$.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> For AR(2), show that the stationarity region in the $(\phi_1, \phi_2)$ plane is triangular with vertices at $(2, -1)$, $(-2, -1)$, and $(0, 1)$.</summary>

<div class="answer">
<strong>Answer:</strong> The three stationarity conditions $\phi_1 + \phi_2 < 1$, $\phi_2 - \phi_1 < 1$, and $|\phi_2| < 1$ define a triangular region.

<strong>Derivation:</strong>
The characteristic equation $1 - \phi_1 z - \phi_2 z^2 = 0$ must have roots outside unit circle.

Setting $z = 1$: $1 - \phi_1 - \phi_2 > 0 \Rightarrow \phi_1 + \phi_2 < 1$
Setting $z = -1$: $1 + \phi_1 - \phi_2 > 0 \Rightarrow \phi_2 - \phi_1 < 1$

For complex roots, discriminant analysis gives: $|\phi_2| < 1$

Boundary lines:
- $\phi_1 + \phi_2 = 1$ (passes through $(2, -1)$ and $(0, 1)$)
- $\phi_2 - \phi_1 = 1$ (passes through $(-2, -1)$ and $(0, 1)$)
- $\phi_2 = -1$ (connects $(2, -1)$ and $(-2, -1)$)

<div class="pitfall">
<strong>Common pitfall:</strong> Checking only one condition. All three must hold simultaneously. A model can satisfy two conditions but fail the third and still be non-stationary.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You fit an AR(3) model and get estimates $\hat{\phi}_1 = 0.5$, $\hat{\phi}_2 = 0.3$, $\hat{\phi}_3 = 0.25$. The residual ACF shows a significant spike at lag 1. What might be wrong?</summary>

<div class="answer">
<strong>Answer:</strong> A significant residual autocorrelation at lag 1 indicates model misspecification. Possible causes:
1. MA component is needed (ARMA instead of pure AR)
2. Structural break in the data
3. Outliers affecting estimation
4. Non-stationarity not fully addressed

<strong>Diagnostic steps:</strong>
1. Perform Ljung-Box test on residuals
2. Try fitting ARMA(3,1) or ARMA(3,2)
3. Plot residuals over time to check for patterns
4. Check for outliers or level shifts
5. Verify original series was stationary

**Key insight:** Pure AR residuals should be white noise. Significant residual autocorrelation means the model hasn't captured all temporal dependence.

<div class="pitfall">
<strong>Common pitfall:</strong> Adding more AR lags to fix residual correlation. Sometimes an MA term is more parsimonious. Compare AIC between AR(4), AR(5) and ARMA(3,1).
</div>
</div>
</details>

## References

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 3.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 3.
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 3.
4. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 2.
