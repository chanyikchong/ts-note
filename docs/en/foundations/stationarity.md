# Stationarity

<div class="interview-summary">
<strong>Interview Summary:</strong> Stationarity is the foundational assumption for most classical time series models. A stationary process has constant mean, constant variance, and autocovariance that depends only on lag, not time. Weak (covariance) stationarity is usually sufficient. Test with ADF, KPSS, or PP tests. Non-stationary series can often be made stationary through differencing.
</div>

## Core Definitions

**Strict (Strong) Stationarity**: A process $\{X_t\}$ is strictly stationary if the joint distribution of $(X_{t_1}, X_{t_2}, \ldots, X_{t_k})$ is identical to $(X_{t_1+h}, X_{t_2+h}, \ldots, X_{t_k+h})$ for all $k$, all time points $t_1, \ldots, t_k$, and all shifts $h$.

**Weak (Covariance/Second-Order) Stationarity**: A process is weakly stationary if:

1. $E[X_t] = \mu$ (constant mean, finite)
2. $\text{Var}(X_t) = \sigma^2 < \infty$ (constant variance, finite)
3. $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$ (autocovariance depends only on lag $h$)

**Ergodicity** (high-level): An ergodic process allows time averages to converge to ensemble averages. This justifies estimating population parameters from a single realization. Most stationary processes encountered in practice are ergodic.

**Trend Stationarity vs. Difference Stationarity**:

- **Trend stationary**: $X_t = \mu_t + Y_t$ where $Y_t$ is stationary; remove trend by regression
- **Difference stationary**: $\Delta X_t = X_t - X_{t-1}$ is stationary; remove unit root by differencing

## Math and Derivations

### Autocovariance Function

For a weakly stationary process:

$$\gamma(h) = \text{Cov}(X_t, X_{t+h}) = E[(X_t - \mu)(X_{t+h} - \mu)]$$

Properties:
- $\gamma(0) = \text{Var}(X_t) = \sigma^2$
- $\gamma(h) = \gamma(-h)$ (symmetry)
- $|\gamma(h)| \leq \gamma(0)$ (Cauchy-Schwarz)

### Autocorrelation Function (ACF)

$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\text{Cov}(X_t, X_{t+h})}{\text{Var}(X_t)}$$

Properties:
- $\rho(0) = 1$
- $|\rho(h)| \leq 1$
- $\rho(h) = \rho(-h)$

### Unit Root and Integration

A process has a unit root if $(1-L)X_t$ is stationary where $L$ is the lag operator ($LX_t = X_{t-1}$).

**Random Walk** (unit root example):
$$X_t = X_{t-1} + \epsilon_t$$

This is non-stationary: $\text{Var}(X_t) = t\sigma^2_\epsilon \to \infty$.

After differencing: $\Delta X_t = \epsilon_t$ which is stationary.

### Augmented Dickey-Fuller (ADF) Test

Tests for unit root. Model:
$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

- $H_0$: $\gamma = 0$ (unit root exists, non-stationary)
- $H_1$: $\gamma < 0$ (no unit root, stationary)

**KPSS Test** (complementary):
- $H_0$: Series is stationary
- $H_1$: Series has a unit root

Use both ADF and KPSS for robust conclusions.

## Algorithm/Model Sketch

**Testing for Stationarity:**

```
1. Visual inspection: plot series, look for trends/changing variance
2. ACF plot: stationary series have ACF that decays to zero
3. ADF test: reject H0 → stationary
4. KPSS test: fail to reject H0 → stationary
5. If non-stationary:
   - Try differencing (for unit root)
   - Try detrending (for trend stationarity)
   - Check if seasonal differencing needed
```

**Making a Series Stationary:**

| Symptom | Solution |
|---------|----------|
| Trend (linear) | First difference or detrend |
| Trend (quadratic) | Second difference |
| Seasonality | Seasonal difference |
| Changing variance | Log transform, then difference |
| Both trend and seasonality | Combine transformations |

## Common Pitfalls

1. **Confusing strict and weak stationarity**: Weak is usually sufficient for ARIMA modeling. Strict is rarely tested directly.

2. **Over-differencing**: Differencing a stationary series introduces unnecessary MA structure. Check ACF—if it's already decaying, don't difference.

3. **Ignoring structural breaks**: A series with a structural break may appear non-stationary but differencing won't help. Consider regime-switching models.

4. **Misinterpreting ADF p-values**: ADF tests for unit root, not stationarity. Low p-value rejects unit root (suggests stationarity). Also use KPSS for confirmation.

5. **Neglecting variance stationarity**: A series can have constant mean but changing variance (heteroskedasticity). Consider GARCH models or transformations.

6. **Seasonal unit roots**: Standard ADF doesn't detect seasonal unit roots. Use HEGY test or seasonal differencing.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

# Generate random walk (non-stationary)
np.random.seed(42)
random_walk = np.cumsum(np.random.randn(200))

# Generate stationary AR(1)
ar1 = np.zeros(200)
for t in range(1, 200):
    ar1[t] = 0.7 * ar1[t-1] + np.random.randn()

# ADF test
adf_rw = adfuller(random_walk)
print(f"Random Walk ADF p-value: {adf_rw[1]:.4f}")  # High → non-stationary

adf_ar = adfuller(ar1)
print(f"AR(1) ADF p-value: {adf_ar[1]:.4f}")  # Low → stationary

# First difference of random walk
diff_rw = np.diff(random_walk)
adf_diff = adfuller(diff_rw)
print(f"Differenced RW ADF p-value: {adf_diff[1]:.4f}")  # Low → stationary
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the difference between strict and weak stationarity? When is weak stationarity sufficient?</summary>

<div class="answer">
<strong>Answer:</strong> Strict stationarity requires the entire joint distribution to be invariant to time shifts. Weak stationarity only requires constant mean, constant variance, and autocovariance depending only on lag.

<strong>Explanation:</strong> Weak stationarity is sufficient for ARIMA-type models because these models only use first and second moments (mean and covariances). The full distributional properties aren't needed for parameter estimation or forecasting.

**Key point:** If a process is strictly stationary with finite second moments, it is also weakly stationary. The reverse is not always true.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming weak stationarity implies Gaussianity. A weakly stationary process can have any marginal distribution—it only constrains the first two moments.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Explain the difference between trend stationarity and difference stationarity. How do you handle each?</summary>

<div class="answer">
<strong>Answer:</strong> Trend stationary: the series has a deterministic trend; subtract the trend to get a stationary residual. Difference stationary: the series has a stochastic trend (unit root); differencing removes the non-stationarity.

<strong>Explanation:</strong>
- Trend stationary: $X_t = \alpha + \beta t + Y_t$ where $Y_t$ is stationary. Fit trend and subtract.
- Difference stationary: $X_t = X_{t-1} + \epsilon_t$. First difference: $\Delta X_t = \epsilon_t$.

Applying the wrong transformation is inefficient—differencing a trend-stationary series adds MA(1) structure; detrending a difference-stationary series leaves autocorrelation.

<div class="pitfall">
<strong>Common pitfall:</strong> Using differencing for everything. Always plot the series and consider whether the trend looks deterministic or stochastic.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Show that for a random walk $X_t = X_{t-1} + \epsilon_t$, the variance grows linearly with time.</summary>

<div class="answer">
<strong>Answer:</strong> $\text{Var}(X_t) = t \cdot \sigma^2_\epsilon$

<strong>Derivation:</strong>
Starting from $X_0 = 0$:
$$X_t = \sum_{i=1}^{t} \epsilon_i$$

Since $\epsilon_i$ are independent with variance $\sigma^2_\epsilon$:
$$\text{Var}(X_t) = \text{Var}\left(\sum_{i=1}^{t} \epsilon_i\right) = \sum_{i=1}^{t} \text{Var}(\epsilon_i) = t \cdot \sigma^2_\epsilon$$

This shows variance grows without bound, violating weak stationarity.

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting that cumulative sums of stationary processes are generally non-stationary. Integration (summation) and differentiation have opposite effects on stationarity.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Derive the autocorrelation function for an AR(1) process $X_t = \phi X_{t-1} + \epsilon_t$ where $|\phi| < 1$.</summary>

<div class="answer">
<strong>Answer:</strong> $\rho(h) = \phi^{|h|}$

<strong>Derivation:</strong>
For stationarity, multiply both sides by $X_{t-h}$ and take expectations:
$$E[X_t X_{t-h}] = \phi E[X_{t-1} X_{t-h}] + E[\epsilon_t X_{t-h}]$$

Since $\epsilon_t$ is uncorrelated with past values:
$$\gamma(h) = \phi \gamma(h-1)$$ for $h \geq 1$

This is a first-order recurrence with solution:
$$\gamma(h) = \phi^h \gamma(0)$$

Therefore:
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \phi^h$$

For $h < 0$, use symmetry: $\rho(h) = \phi^{|h|}$

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting the stationarity condition $|\phi| < 1$. If $|\phi| \geq 1$, the process explodes and has no finite variance.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You run an ADF test and get p-value = 0.08, and a KPSS test with p-value = 0.03. What do you conclude? What should you do?</summary>

<div class="answer">
<strong>Answer:</strong> The results are contradictory. ADF suggests possible stationarity (p = 0.08 is borderline), but KPSS rejects stationarity (p = 0.03). This often indicates the series is near-unit-root or has a structural break.

<strong>Recommended actions:</strong>
1. Plot the series and ACF to visually inspect
2. Try differencing and re-test
3. Check for structural breaks (Chow test, CUSUM)
4. Consider that the series may be fractionally integrated
5. Use domain knowledge—is non-stationarity expected?

**Key equation:** For KPSS, low p-value rejects stationarity. For ADF, low p-value rejects unit root (supports stationarity).

<div class="pitfall">
<strong>Common pitfall:</strong> Relying on a single test. ADF has low power against near-unit-root alternatives. KPSS can reject stationarity due to serial correlation. Always use multiple approaches and visual inspection.
</div>
</div>
</details>

## References

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapters 3, 17.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 1.
3. Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *JASA*, 74(366), 427-431.
4. Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity. *Journal of Econometrics*, 54(1-3), 159-178.
