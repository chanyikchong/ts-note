# Model Identification

<div class="interview-summary">
<strong>Interview Summary:</strong> Model identification determines (p,d,q) orders using ACF/PACF patterns, unit root tests, and information criteria. AR(p): PACF cuts off at p. MA(q): ACF cuts off at q. ARMA: both tail off. Use ADF/KPSS for d. For complex cases, use AIC/BIC to compare candidates. Always validate with residual diagnostics.
</div>

## Core Definitions

**Model Identification**: The process of determining:
1. Transformation needed (log, Box-Cox)
2. Order of differencing (d, D)
3. AR order (p, P)
4. MA order (q, Q)

**ACF/PACF Patterns for Pure Models:**

| Model | ACF | PACF |
|-------|-----|------|
| AR(p) | Tails off (exponential/sinusoidal decay) | Cuts off after lag p |
| MA(q) | Cuts off after lag q | Tails off (exponential/sinusoidal decay) |
| ARMA(p,q) | Tails off | Tails off |
| Non-stationary | Very slow decay | Large spike at lag 1 |
| White noise | All near zero | All near zero |

**Significance Threshold**: Under white noise, $\hat{\rho}(h) \sim N(0, 1/n)$, so use $\pm 1.96/\sqrt{n}$ for 95% bands.

## Math and Derivations

### Information Criteria

**AIC (Akaike Information Criterion)**:
$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

where $\hat{L}$ is maximum likelihood and $k$ is number of parameters.

For ARIMA with Gaussian errors:
$$\text{AIC} = n\ln(\hat{\sigma}^2) + 2(p+q+1)$$

**BIC (Bayesian/Schwarz Information Criterion)**:
$$\text{BIC} = -2\ln(\hat{L}) + k\ln(n)$$

BIC penalizes complexity more heavily than AIC.

**AICc (Corrected AIC)**:
$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

Preferred for small samples.

### Extended ACF (EACF)

The EACF method iteratively removes AR structure to identify MA order:

1. Fit AR(0), AR(1), AR(2), ... up to max
2. For each AR(j), compute ACF of residuals
3. Create table: rows = AR order, columns = MA order
4. "O" indicates insignificant, "X" indicates significant
5. Top-left corner of "O" triangle suggests (p,q)

### Unit Root Testing Strategy

**Combined ADF + KPSS approach:**

| ADF result | KPSS result | Conclusion |
|------------|-------------|------------|
| Reject (p < 0.05) | Fail to reject | Stationary (d=0) |
| Fail to reject | Reject | Non-stationary (d≥1) |
| Reject | Reject | Possible structural break |
| Fail to reject | Fail to reject | Inconclusive; try differencing |

### Ljung-Box Test for Residual Autocorrelation

$$Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$

Under null (white noise residuals): $Q(m) \sim \chi^2_{m-p-q}$

Reject if Q is large (residuals are not white noise).

## Algorithm/Model Sketch

**Complete Identification Procedure:**

```
STEP 1: PRELIMINARY ANALYSIS
- Plot the series
- Check for obvious trend, seasonality, outliers
- Apply transformation if variance changes with level (log, sqrt)

STEP 2: DETERMINE DIFFERENCING ORDER
- ADF test: if p > 0.05, difference
- KPSS test: if p < 0.05, difference
- After differencing, re-test
- Usually d ≤ 2; rarely d > 2

STEP 3: EXAMINE ACF/PACF OF STATIONARY SERIES
- ACF cuts off at lag q → try MA(q)
- PACF cuts off at lag p → try AR(p)
- Both tail off → try ARMA

STEP 4: FIT CANDIDATE MODELS
- Start simple: AR(1), MA(1), ARMA(1,1)
- Add complexity as needed
- Compare AIC/BIC

STEP 5: DIAGNOSTIC CHECKING
- Residual ACF/PACF (should be white noise)
- Ljung-Box test
- Residual normality (Q-Q plot)
- Parameter significance

STEP 6: SELECT FINAL MODEL
- Lowest AIC/BIC among adequate models
- Parsimony when AIC/BIC are close
- Good residual diagnostics
```

**Auto-Selection Tools:**
- `auto.arima()` in R (forecast package)
- `pmdarima.auto_arima()` in Python
- Uses stepwise search with AIC

## Common Pitfalls

1. **Mechanical ACF/PACF reading**: Patterns aren't always clean. Real data is noisy. Consider multiple interpretations.

2. **Ignoring parsimony**: If ARMA(1,1) and ARMA(2,1) have similar AIC, prefer simpler model.

3. **Over-relying on automated selection**: `auto.arima` is a good start but may miss important features. Always verify manually.

4. **Forgetting seasonal patterns**: Check ACF at seasonal lags (12, 24, ... for monthly). Standard ARIMA won't capture these.

5. **Ignoring residual diagnostics**: A model with lowest AIC can still have autocorrelated residuals. Always check.

6. **Sample size issues**: With small samples (n < 50), ACF/PACF estimates are unreliable. Use simpler models and be conservative.

## Mini Example

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# Generate ARIMA(2,1,1) data for identification exercise
np.random.seed(42)
n = 250
phi1, phi2, theta = 0.5, -0.2, 0.3
eps = np.random.randn(n + 3)

# Differenced series as ARMA(2,1)
dX = np.zeros(n)
for t in range(2, n):
    dX[t] = phi1*dX[t-1] + phi2*dX[t-2] + eps[t+1] + theta*eps[t]

X = np.cumsum(dX) + 50  # Integrate and add level

# Step 1: Test stationarity
print("Step 1: Stationarity Tests")
print(f"ADF p-value (levels): {adfuller(X)[1]:.4f}")
print(f"KPSS p-value (levels): {kpss(X, regression='c')[1]:.4f}")

# Step 2: Difference and re-test
dX_obs = np.diff(X)
print(f"\nADF p-value (differenced): {adfuller(dX_obs)[1]:.4f}")
print(f"KPSS p-value (differenced): {kpss(dX_obs, regression='c')[1]:.4f}")

# Step 3: ACF/PACF of differenced series
print("\nStep 3: ACF/PACF")
acf_vals = acf(dX_obs, nlags=10)
pacf_vals = pacf(dX_obs, nlags=10)
print(f"ACF: {np.round(acf_vals[1:6], 3)}")
print(f"PACF: {np.round(pacf_vals[1:6], 3)}")

# Step 4: Compare candidate models
print("\nStep 4: Model Comparison")
models = {
    'ARIMA(1,1,1)': (1,1,1),
    'ARIMA(2,1,0)': (2,1,0),
    'ARIMA(2,1,1)': (2,1,1),
    'ARIMA(1,1,2)': (1,1,2),
}

for name, order in models.items():
    try:
        model = ARIMA(X, order=order).fit()
        lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
        print(f"{name}: AIC={model.aic:.1f}, BIC={model.bic:.1f}, LB p-value={lb['lb_pvalue'].values[0]:.3f}")
    except:
        print(f"{name}: Failed to converge")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why do we say ACF "cuts off" for MA(q) but "tails off" for AR(p)?</summary>

<div class="answer">
<strong>Answer:</strong>
- MA(q): $X_t$ only involves $\epsilon_t, \epsilon_{t-1}, \ldots, \epsilon_{t-q}$. Beyond lag q, $X_t$ and $X_{t-h}$ share no common $\epsilon$ terms, so $\gamma(h) = 0$ exactly.
- AR(p): $X_t$ depends on all past values through the recursive structure. Even though only p lags appear explicitly, the chain of dependencies means $X_t$ correlates with $X_{t-h}$ for all $h$.

<strong>Key insight:</strong> AR has infinite memory (gradually decaying); MA has finite memory (exactly q lags).

<div class="pitfall">
<strong>Common pitfall:</strong> In practice, "cutoff" doesn't mean exactly zero — sample ACF will have small nonzero values beyond q due to estimation error. Look for sharp drop versus gradual decay.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What's the difference between AIC and BIC? When would you prefer one over the other?</summary>

<div class="answer">
<strong>Answer:</strong>
- AIC: $-2\ln(L) + 2k$; penalizes parameters lightly
- BIC: $-2\ln(L) + k\ln(n)$; penalty grows with sample size

**BIC preference situations:**
- Large sample sizes (BIC penalty more appropriate)
- When true model is among candidates (BIC is consistent)
- For inference/explanation (simpler models)

**AIC preference situations:**
- Forecasting focus (AIC optimizes prediction)
- Small samples (use AICc)
- When true model may be complex

<div class="pitfall">
<strong>Common pitfall:</strong> Blindly using one criterion. For forecasting, cross-validation often beats both AIC and BIC. Use information criteria as guides, not final arbiters.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Show that for a white noise process, the sample ACF has approximate variance $1/n$.</summary>

<div class="answer">
<strong>Answer:</strong> For white noise with $\rho(h) = 0$ for $h \neq 0$:

<strong>Derivation (Bartlett's approximation):</strong>

For large n, sample ACF is approximately normal:
$$\hat{\rho}(h) \sim N(\rho(h), V_h/n)$$

where $V_h = \sum_{j=-\infty}^{\infty} [\rho(j)\rho(j+h) + \rho(j+h)\rho(j-h) - 2\rho(h)\rho(j)\rho(j+h)]$

For white noise ($\rho(j) = 0$ for $j \neq 0$):
$$V_h = 1 \text{ for all } h \neq 0$$

Therefore: $\text{Var}(\hat{\rho}(h)) \approx 1/n$

**95% confidence interval:** $\hat{\rho}(h) \pm 1.96/\sqrt{n}$

<div class="pitfall">
<strong>Common pitfall:</strong> This formula only holds under white noise. For non-white-noise series, the variance is different (Bartlett's formula gives different expressions).
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> The Ljung-Box test statistic is $Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$. Why is there a $(n-k)$ denominator?</summary>

<div class="answer">
<strong>Answer:</strong> The $(n-k)$ term is a small-sample correction. At lag $k$, only $n-k$ pairs of observations contribute to $\hat{\rho}(k)$, making the estimate less precise.

<strong>Explanation:</strong>
The original Box-Pierce statistic used $Q' = n\sum \hat{\rho}_k^2$. Ljung and Box modified it because:
1. $\text{Var}(\hat{\rho}(k)) \approx (n-k)/n^2$ rather than $1/n$
2. The correction $(n+2)/(n-k)$ improves the chi-squared approximation in finite samples

Without correction, the test is undersized (rejects less than it should), missing autocorrelation.

<div class="pitfall">
<strong>Common pitfall:</strong> Choosing m too large. If $m > n/4$, the test loses power. Common choices: $m = 10$ for non-seasonal, $m = 2s$ for seasonal data.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You examine ACF/PACF and see: ACF decays slowly over first 3 lags then drops; PACF has spikes at lags 1, 2, 3 then drops. AIC favors ARIMA(3,0,0) but residuals show significant ACF at lag 1. What's your next step?</summary>

<div class="answer">
<strong>Answer:</strong> The significant residual ACF at lag 1 indicates model inadequacy despite good AIC. Try:

1. **Add MA(1)**: Try ARIMA(3,0,1) — MA term may capture the remaining lag-1 correlation
2. **Check for near-redundancy**: If AR(3) coefficients are close to forming MA factor, simplify
3. **Consider ARIMA(2,0,1)**: Sometimes mixed model is more parsimonious
4. **Check for outliers**: Single outliers can cause lag-1 residual correlation
5. **Re-examine stationarity**: Maybe $d=1$ differencing is needed

**Key principle:** A model isn't adequate until residuals are white noise. AIC is necessary but not sufficient — must pass diagnostic checks.

<div class="pitfall">
<strong>Common pitfall:</strong> Accepting a model just because it has lowest AIC. Always verify residual ACF is within bands at all lags, especially early ones.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapters 6-8.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 9.
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. Chapter 2.
4. Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.
