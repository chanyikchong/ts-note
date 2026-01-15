# Residual Diagnostics

<div class="interview-summary">
<strong>Interview Summary:</strong> Residuals should be white noise if model is adequate. Check via: ACF plot (no significant spikes), Ljung-Box test (p > 0.05), normality (Q-Q plot), and homoskedasticity (constant variance). Patterns in residuals indicate model inadequacy: autocorrelation suggests missing AR/MA terms; changing variance suggests GARCH; trends suggest wrong differencing.
</div>

## Core Definitions

**Residuals:** $e_t = y_t - \hat{y}_{t|t-1}$ (one-step-ahead prediction errors)

**Standardized Residuals:** $z_t = e_t / \hat{\sigma}$ (should be approximately N(0,1))

**White Noise Properties:**
- $E[e_t] = 0$
- $\text{Var}(e_t) = \sigma^2$ (constant)
- $\text{Cov}(e_t, e_{t-k}) = 0$ for $k \neq 0$

## Math and Derivations

### Ljung-Box Test

Tests whether autocorrelations are jointly zero.

$$Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$

Under H₀ (white noise): $Q(m) \sim \chi^2_{m-p-q}$ (adjusted for estimated parameters)

**Decision:** Reject H₀ if Q > critical value (or p < α)

### Jarque-Bera Normality Test

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

where S = skewness, K = kurtosis.

Under H₀ (normality): $JB \sim \chi^2_2$

### ARCH-LM Test for Heteroskedasticity

Test if $e_t^2$ depends on past squared residuals:
$$e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + \cdots + \alpha_p e_{t-p}^2 + v_t$$

Test statistic: $nR^2 \sim \chi^2_p$ under H₀ (homoskedasticity)

### Runs Test for Randomness

Counts runs (consecutive same-sign residuals). Too few runs suggests autocorrelation; too many suggests over-differencing.

## Algorithm/Model Sketch

**Diagnostic Checklist:**

```
1. MEAN ZERO
   □ Mean of residuals ≈ 0
   □ Plot residuals over time: no trend

2. NO AUTOCORRELATION
   □ ACF plot: all spikes within ±1.96/√n bands
   □ PACF plot: no patterns
   □ Ljung-Box test: p > 0.05 at multiple lags

3. CONSTANT VARIANCE
   □ Plot residuals vs time: no fanning/clustering
   □ Plot residuals vs fitted: no pattern
   □ ARCH test: p > 0.05

4. NORMALITY (less critical)
   □ Histogram: roughly bell-shaped
   □ Q-Q plot: points on diagonal
   □ Jarque-Bera: p > 0.05

5. NO OUTLIERS
   □ |standardized residuals| < 3 mostly
   □ Check any points > 3 for data issues
```

**Interpretation of Violations:**

| Violation | Interpretation | Fix |
|-----------|---------------|-----|
| Lag 1 ACF spike | Missing MA(1) | Add MA term |
| Lag 1 PACF spike | Missing AR(1) | Add AR term |
| Seasonal spikes | Missing seasonal | Add seasonal terms |
| Slow ACF decay | Under-differencing | Increase d |
| Negative ACF at lag 1 | Over-differencing | Decrease d |
| Changing variance | Heteroskedasticity | GARCH, log-transform |
| Non-normality | Heavy tails | Robust methods, outlier treatment |

## Common Pitfalls

1. **Over-testing**: With many lags, some will be significant by chance. Focus on early lags and patterns.

2. **Ignoring degrees of freedom**: Ljung-Box df = m - p - q, not m. Wrong df gives wrong p-values.

3. **Choosing m poorly**: Too small m misses long-range dependence; too large has low power. Rule: m ≈ min(10, n/5).

4. **Normality obsession**: Non-normality is often acceptable. Autocorrelation is the critical check.

5. **Missing patterns at seasonal lags**: Always check ACF at lags 12, 24 (monthly), 7, 14 (daily), etc.

6. **Confusing residuals and innovations**: For MA models, residuals ≠ true innovations. Some autocorrelation is expected in finite samples.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

# Generate data and fit intentionally wrong model
np.random.seed(42)
n = 200
# True: ARMA(1,1)
phi, theta = 0.7, 0.4
eps = np.random.randn(n + 1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

# Fit AR(1) only (missing MA term)
model_wrong = ARIMA(y, order=(1, 0, 0)).fit()
resid = model_wrong.resid

print("=== Residual Diagnostics ===\n")

# 1. Mean
print(f"1. Mean: {np.mean(resid):.4f} (should be ≈ 0)")

# 2. Autocorrelation
print("\n2. Autocorrelation:")
lb_test = acorr_ljungbox(resid, lags=[5, 10, 15], return_df=True)
print(lb_test)

# 3. Normality
jb_stat, jb_p = stats.jarque_bera(resid)
print(f"\n3. Normality (Jarque-Bera): stat={jb_stat:.2f}, p={jb_p:.4f}")

# 4. Check ACF
acf_vals = np.correlate(resid, resid, mode='full')
acf_vals = acf_vals[len(acf_vals)//2:] / acf_vals[len(acf_vals)//2]
print(f"\n4. ACF at lag 1: {acf_vals[1]:.3f} (significant if |.| > {1.96/np.sqrt(n):.3f})")

# Compare with correct model
model_correct = ARIMA(y, order=(1, 0, 1)).fit()
resid_correct = model_correct.resid
lb_correct = acorr_ljungbox(resid_correct, lags=[5, 10, 15], return_df=True)
print("\n=== Correct Model (ARMA(1,1)) ===")
print(lb_correct)
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why is checking residual autocorrelation more important than checking normality?</summary>

<div class="answer">
<strong>Answer:</strong>

**Autocorrelation matters more because:**
1. **Biased forecasts**: Residual autocorrelation means systematic patterns remain unexploited
2. **Invalid inference**: Standard errors and confidence intervals assume independence
3. **Model inadequacy**: Autocorrelation directly indicates missing structure
4. **Fixable**: Can add AR/MA terms to remove autocorrelation

**Normality is less critical because:**
1. **Robust methods exist**: Point forecasts don't require normality
2. **CLT helps**: Averages become normal even if residuals aren't
3. **Only affects intervals**: Normality matters mainly for prediction intervals
4. **Often ignorable**: Heavy tails don't bias forecasts, just widen intervals

<div class="pitfall">
<strong>Common pitfall:</strong> Spending effort on normality transformations while ignoring autocorrelation. Fix autocorrelation first; normality can often be ignored.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What does a significant negative spike at lag 1 in the residual ACF suggest?</summary>

<div class="answer">
<strong>Answer:</strong> Likely **over-differencing**.

**Explanation:**
Differencing a stationary series introduces MA(1) with θ ≈ -1:
$$(1-L)y_t = \epsilon_t - \epsilon_{t-1} \text{ approximately}$$

This has ACF: $\rho(1) = -1/(1+1) = -0.5$

So large negative lag-1 ACF (around -0.3 to -0.5) suggests you differenced a series that was already stationary.

**Action:**
1. Re-test original series for stationarity
2. Try model without differencing
3. Compare AIC between d=0 and d=1

<div class="pitfall">
<strong>Common pitfall:</strong> Reflexively differencing because it's "standard procedure." Check stationarity tests and residuals before and after differencing.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> The Ljung-Box test uses degrees of freedom m-p-q. Why subtract p+q?</summary>

<div class="answer">
<strong>Answer:</strong> We subtract estimated parameters to account for their effect on residuals.

**Explanation:**
For a fitted ARMA(p,q), the residuals $e_t = y_t - \hat{y}_{t|t-1}$ are computed using estimated $\hat{\phi}$, $\hat{\theta}$.

The estimation process uses up information from the data, reducing effective degrees of freedom. Specifically:
- p AR parameters constrain p lagged autocorrelations
- q MA parameters constrain q lagged autocorrelations

Under H₀, the test statistic:
$$Q(m) \sim \chi^2_{m-p-q}$$

not $\chi^2_m$. Using m degrees of freedom would reject too often (test is oversized).

<div class="pitfall">
<strong>Common pitfall:</strong> Software may not adjust df automatically. Verify that p and q are subtracted; otherwise p-values are wrong.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How do you interpret the ARCH-LM test for residuals?</summary>

<div class="answer">
<strong>Answer:</strong> ARCH-LM tests for conditional heteroskedasticity — whether variance depends on past volatility.

**Procedure:**
1. Compute squared residuals $e_t^2$
2. Regress: $e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + \cdots + \alpha_p e_{t-p}^2$
3. Test: $H_0$: all $\alpha_i = 0$ (homoskedasticity)

**Test statistic:** $nR^2 \sim \chi^2_p$

**Interpretation:**
- p < 0.05: Evidence of ARCH effects; variance clusters
- p > 0.05: No evidence; constant variance OK

**If significant:**
- Consider GARCH model
- Or variance-stabilizing transform (log)
- Prediction intervals need adjustment

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring ARCH effects leads to prediction intervals that are too narrow during volatile periods and too wide during calm periods.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You fit ARIMA(1,1,1) and residual diagnostics show: Ljung-Box p=0.02 at lag 10, but p=0.15 at lags 5 and 15. Q-Q plot shows slight heavy tails. What do you conclude?</summary>

<div class="answer">
<strong>Answer:</strong> Model is likely adequate; don't over-interpret the lag-10 result.

**Analysis:**
1. **Ljung-Box at lag 10:** p=0.02 is borderline. But lags 5 and 15 are fine.
   - Could be spurious (multiple testing)
   - Or minor model inadequacy that doesn't matter for forecasting

2. **Heavy tails:** Common in economic/financial data
   - Doesn't invalidate forecasts
   - Affects prediction intervals (may need wider)

**Recommended actions:**
1. Check ACF visually — isolated spike at lag 10 likely noise
2. Compare to simpler models (ARIMA(1,1,0)) — if similar forecasts, prefer simpler
3. For intervals, consider bootstrap or t-distribution
4. Validate on holdout data — ultimate test

**Conclusion:** Accept model unless holdout validation shows problems. Perfect residuals are unrealistic; "good enough" is the standard.

<div class="pitfall">
<strong>Common pitfall:</strong> Chasing perfect diagnostics. Adding parameters to fix one borderline test often causes overfitting. Focus on forecast performance.
</div>
</div>
</details>

## References

1. Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 8.
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. Chapter 2.
4. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
