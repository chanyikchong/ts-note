# Autocorrelation and Partial Autocorrelation

<div class="interview-summary">
<strong>Interview Summary:</strong> ACF measures correlation between a series and its lagged values. PACF measures correlation at lag $k$ after removing effects of intermediate lags. ACF/PACF patterns identify AR/MA orders: AR(p) has PACF cutoff at lag p; MA(q) has ACF cutoff at lag q. Sample ACF/PACF have approximate standard error $1/\sqrt{n}$ under white noise.
</div>

## Core Definitions

**Autocovariance Function (ACVF)**: For a stationary process with mean $\mu$:
$$\gamma(h) = \text{Cov}(X_t, X_{t+h}) = E[(X_t - \mu)(X_{t+h} - \mu)]$$

**Autocorrelation Function (ACF)**: Normalized autocovariance:
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \text{Corr}(X_t, X_{t+h})$$

**Partial Autocorrelation Function (PACF)**: Correlation between $X_t$ and $X_{t+h}$ after removing the linear dependence on $X_{t+1}, \ldots, X_{t+h-1}$:
$$\phi_{hh} = \text{Corr}(X_t - \hat{X}_t, X_{t+h} - \hat{X}_{t+h})$$

where $\hat{X}_t$ and $\hat{X}_{t+h}$ are best linear predictors from intermediate values.

**Sample ACF**: Estimated from data:
$$\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)} = \frac{\sum_{t=1}^{n-h}(X_t - \bar{X})(X_{t+h} - \bar{X})}{\sum_{t=1}^{n}(X_t - \bar{X})^2}$$

## Math and Derivations

### Properties of ACF

For any stationary process:

1. $\rho(0) = 1$
2. $\rho(h) = \rho(-h)$ (symmetry)
3. $|\rho(h)| \leq 1$ for all $h$
4. $\rho(h)$ is positive semi-definite: for any $a_1, \ldots, a_n$:
   $$\sum_{i=1}^{n}\sum_{j=1}^{n} a_i a_j \rho(i-j) \geq 0$$

### PACF via Yule-Walker Equations

The PACF at lag $k$ is the last coefficient $\phi_{kk}$ in the AR(k) regression:
$$X_t = \phi_{k1}X_{t-1} + \phi_{k2}X_{t-2} + \cdots + \phi_{kk}X_{t-k} + \epsilon_t$$

Yule-Walker equations in matrix form:
$$\begin{pmatrix} 1 & \rho(1) & \cdots & \rho(k-1) \\ \rho(1) & 1 & \cdots & \rho(k-2) \\ \vdots & & \ddots & \vdots \\ \rho(k-1) & \cdots & \rho(1) & 1 \end{pmatrix} \begin{pmatrix} \phi_{k1} \\ \phi_{k2} \\ \vdots \\ \phi_{kk} \end{pmatrix} = \begin{pmatrix} \rho(1) \\ \rho(2) \\ \vdots \\ \rho(k) \end{pmatrix}$$

### ACF of AR(1): $X_t = \phi X_{t-1} + \epsilon_t$

$$\rho(h) = \phi^{|h|}$$

ACF decays exponentially (geometrically) for $|\phi| < 1$.

### ACF of MA(1): $X_t = \epsilon_t + \theta\epsilon_{t-1}$

$$\rho(1) = \frac{\theta}{1+\theta^2}, \quad \rho(h) = 0 \text{ for } h > 1$$

ACF cuts off after lag 1.

### ACF of MA(q)

$$\rho(h) = 0 \text{ for } h > q$$

### PACF of AR(p)

$$\phi_{hh} = 0 \text{ for } h > p$$

### Variance of Sample ACF

Under the null hypothesis that the true process is white noise:
$$\text{Var}(\hat{\rho}(h)) \approx \frac{1}{n}$$

So approximate 95% confidence bands are $\pm 1.96/\sqrt{n}$.

**Bartlett's formula** (for MA(q) process):
$$\text{Var}(\hat{\rho}(h)) \approx \frac{1}{n}\left(1 + 2\sum_{k=1}^{q}\rho(k)^2\right) \text{ for } h > q$$

## Algorithm/Model Sketch

**Using ACF/PACF for Model Identification:**

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| AR(p) | Exponential/sinusoidal decay | Cuts off after lag p | AR(p) |
| MA(q) | Cuts off after lag q | Exponential/sinusoidal decay | MA(q) |
| ARMA(p,q) | Tails off | Tails off | ARMA(p,q) |
| White noise | All near zero | All near zero | No model needed |
| Non-stationary | Very slow decay | Large spike at lag 1 | Difference first |

**Interpretation procedure:**

```
1. Plot series - check for stationarity
2. If non-stationary, difference until stationary
3. Compute sample ACF and PACF
4. Check for significant spikes (outside ±1.96/√n bands)
5. Identify patterns:
   - ACF cuts off, PACF decays → MA(q) where q = cutoff lag
   - PACF cuts off, ACF decays → AR(p) where p = cutoff lag
   - Both decay → ARMA (use information criteria)
6. Fit candidate models
7. Check residual ACF/PACF (should be white noise)
```

## Common Pitfalls

1. **Ignoring confidence bands**: Not all spikes are significant. Use $\pm 1.96/\sqrt{n}$ bands and expect ~5% of spikes to exceed by chance.

2. **Confusing "cuts off" vs "tails off"**: Cutoff means abrupt drop to zero after lag q. Tails off means gradual decay. This distinction determines AR vs MA.

3. **Applying ACF/PACF to non-stationary data**: Results are meaningless for non-stationary series. Always check stationarity first.

4. **Over-interpreting high-lag correlations**: For small samples, high-lag estimates have high variance. Focus on early lags.

5. **Forgetting seasonal lags**: In seasonal data, check lags at seasonal period (e.g., lag 12 for monthly data with annual seasonality).

6. **Neglecting theoretical ACF/PACF**: When validating models, compare sample functions to theoretical ones, not just residuals.

## Mini Example

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

# Simulate AR(2) process
np.random.seed(42)
ar_params = np.array([1, -0.75, 0.25])  # 1 - 0.75L + 0.25L^2
ma_params = np.array([1])
ar2_process = ArmaProcess(ar_params, ma_params)
ar2_data = ar2_process.generate_sample(nsample=300)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ar2_data, ax=axes[0], lags=20, title='ACF of AR(2)')
plot_pacf(ar2_data, ax=axes[1], lags=20, title='PACF of AR(2)')
plt.tight_layout()
plt.show()

# AR(2): PACF should cut off after lag 2, ACF should decay
# Check significant PACF values
from statsmodels.tsa.stattools import pacf
pacf_values = pacf(ar2_data, nlags=5)
print("PACF values:", pacf_values)
# Expect: significant at lags 1,2; near zero after
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the fundamental difference between ACF and PACF? Why do we need both?</summary>

<div class="answer">
<strong>Answer:</strong> ACF measures total correlation between $X_t$ and $X_{t+h}$, including indirect effects through intermediate lags. PACF measures direct correlation after removing intermediate effects.

<strong>Explanation:</strong> Consider AR(1) with $\phi = 0.8$. The ACF shows $\rho(2) = 0.64$ because $X_t$ and $X_{t+2}$ are correlated through $X_{t+1}$. But the PACF at lag 2 is near zero because once we account for $X_{t+1}$, there's no additional direct relationship.

We need both because:
- ACF identifies MA order (cuts off at lag q)
- PACF identifies AR order (cuts off at lag p)

<div class="pitfall">
<strong>Common pitfall:</strong> Using only ACF for identification. Without PACF, you cannot distinguish AR from MA patterns—both can show decaying ACF, but only AR shows PACF cutoff.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> How would you interpret an ACF that shows a very slow decay with values staying significant past lag 20?</summary>

<div class="answer">
<strong>Answer:</strong> This strongly suggests non-stationarity. A stationary process should have ACF that decays relatively quickly to zero. Very slow decay indicates a unit root or near-unit root.

<strong>Recommended action:</strong>
1. Formally test with ADF/KPSS
2. Difference the series
3. Re-compute ACF after differencing
4. The differenced series should show faster decay

**Key insight:** For a random walk, $\rho(h) \approx 1$ for all $h$ in finite samples because successive values are highly dependent.

<div class="pitfall">
<strong>Common pitfall:</strong> Trying to fit ARMA models to non-stationary data. The resulting parameters will be misleading, and forecasts will be poor. Always ensure stationarity first.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the ACF for MA(1): $X_t = \epsilon_t + \theta\epsilon_{t-1}$.</summary>

<div class="answer">
<strong>Answer:</strong> $\rho(0) = 1$, $\rho(1) = \frac{\theta}{1+\theta^2}$, $\rho(h) = 0$ for $h \geq 2$.

<strong>Derivation:</strong>

**Variance (lag 0):**
$$\gamma(0) = \text{Var}(X_t) = \text{Var}(\epsilon_t + \theta\epsilon_{t-1}) = \sigma^2 + \theta^2\sigma^2 = (1+\theta^2)\sigma^2$$

**Autocovariance at lag 1:**
$$\gamma(1) = \text{Cov}(X_t, X_{t+1}) = \text{Cov}(\epsilon_t + \theta\epsilon_{t-1}, \epsilon_{t+1} + \theta\epsilon_t)$$
$$= \text{Cov}(\epsilon_t, \theta\epsilon_t) = \theta\sigma^2$$

**Autocovariance at lag $h \geq 2$:**
$$\gamma(h) = \text{Cov}(\epsilon_t + \theta\epsilon_{t-1}, \epsilon_{t+h} + \theta\epsilon_{t+h-1}) = 0$$

(No overlapping $\epsilon$ terms when $h \geq 2$)

**ACF:**
$$\rho(1) = \frac{\gamma(1)}{\gamma(0)} = \frac{\theta\sigma^2}{(1+\theta^2)\sigma^2} = \frac{\theta}{1+\theta^2}$$

<div class="pitfall">
<strong>Common pitfall:</strong> Note that $|\rho(1)| \leq 0.5$ for MA(1). The maximum occurs at $\theta = \pm 1$. If you observe $|\hat{\rho}(1)| > 0.5$, it might be AR, not MA.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> For an AR(2) process $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$, show that the PACF is zero for all lags $h > 2$.</summary>

<div class="answer">
<strong>Answer:</strong> For AR(p), the PACF $\phi_{hh} = 0$ for $h > p$ because once $X_{t-1}, \ldots, X_{t-p}$ are included, adding more lags provides no additional predictive information.

<strong>Explanation:</strong>

The AR(2) model states that $X_t$ depends only on $X_{t-1}$ and $X_{t-2}$ (plus noise). Therefore:

For $h = 3$: We regress $X_t$ on $X_{t-1}, X_{t-2}, X_{t-3}$.
- $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t$
- $X_{t-3}$ only affects $X_t$ through $X_{t-2}$ and $X_{t-1}$
- After conditioning on $X_{t-1}$ and $X_{t-2}$, $X_{t-3}$ adds no information
- Thus $\phi_{33} = 0$

By induction, this holds for all $h > 2$.

**Key equation:** PACF at lag $k$ is the coefficient $\phi_{kk}$ in the best linear predictor using exactly $k$ lags. For AR(p), the $k$-th coefficient becomes zero when $k > p$.

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting exact zeros in sample PACF. Due to sampling variability, $\hat{\phi}_{hh}$ will be nonzero but should fall within confidence bands for $h > p$.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You compute the sample ACF for a time series with $n=100$ observations. You see that lags 1, 2, 7, and 15 exceed the 95% confidence bands. What is your interpretation?</summary>

<div class="answer">
<strong>Answer:</strong> With 95% bands and ~15-20 lags tested, expect about 1 spurious significant lag by chance. Lags 1 and 2 are likely real signal. Lag 7 might be real (check for weekly patterns if relevant). Lag 15 is likely spurious.

<strong>Interpretation process:</strong>
1. Focus on early lags (1, 2, 3) — most likely real
2. Consider domain knowledge (lag 7 = weekly? lag 12 = monthly?)
3. Isolated high-lag spikes are often noise
4. Pattern of significant lags matters more than individual spikes
5. Lag 15 with $n=100$ has high variance: $\text{SE} \approx 1/\sqrt{100} = 0.1$, and only ~85 pairs contribute

**Confidence band:** $\pm 1.96/\sqrt{100} = \pm 0.196$

<div class="pitfall">
<strong>Common pitfall:</strong> Treating every significant lag as meaningful. With many lags, false positives occur. Use sequential testing corrections or focus on meaningful patterns, not isolated spikes.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley. Chapter 2.
2. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. Chapter 3.
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapters 2-3.
4. Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time-series. *JRSS B*, 8(1), 27-41.
