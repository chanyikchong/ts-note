# Prediction Intervals

<div class="interview-summary">
<strong>Interview Summary:</strong> Prediction intervals quantify forecast uncertainty, giving a range where future values will likely fall. For ARIMA: PI width grows with horizon due to accumulated uncertainty. Key formula: $\hat{y}_{T+h} \pm z_{\alpha/2}\sigma_h$ where $\sigma_h$ depends on model. Intervals assume normality; bootstrap provides non-parametric alternative.
</div>

## Core Definitions

**Point Forecast:** Single best estimate of future value
$$\hat{y}_{T+h|T} = E[y_{T+h}|y_1,\ldots,y_T]$$

**Prediction Interval:** Range containing future value with probability $(1-\alpha)$
$$[\hat{y}_{T+h|T} - z_{\alpha/2}\sigma_h, \hat{y}_{T+h|T} + z_{\alpha/2}\sigma_h]$$

**Forecast Error:** $e_{T+h|T} = y_{T+h} - \hat{y}_{T+h|T}$

**Forecast Variance:** $\sigma_h^2 = \text{Var}(e_{T+h|T})$

**Coverage Probability:** Proportion of actual values falling within PI (should match nominal level).

## Math and Derivations

### ARMA(p,q) Forecast Variance

The h-step forecast error can be written as:
$$e_{T+h|T} = \sum_{j=0}^{h-1}\psi_j\epsilon_{T+h-j}$$

where $\psi_j$ are MA(∞) coefficients.

Forecast variance:
$$\sigma_h^2 = \sigma_\epsilon^2\sum_{j=0}^{h-1}\psi_j^2$$

### Specific Models

**AR(1):** $y_t = \phi y_{t-1} + \epsilon_t$
$$\psi_j = \phi^j$$
$$\sigma_h^2 = \sigma_\epsilon^2\frac{1-\phi^{2h}}{1-\phi^2}$$

As $h \to \infty$: $\sigma_h^2 \to \sigma_\epsilon^2/(1-\phi^2) = \text{Var}(y_t)$

**MA(1):** $y_t = \epsilon_t + \theta\epsilon_{t-1}$
$$\sigma_1^2 = \sigma_\epsilon^2$$
$$\sigma_h^2 = \sigma_\epsilon^2(1+\theta^2) \text{ for } h \geq 2$$

**Random Walk (ARIMA(0,1,0)):**
$$\sigma_h^2 = h\sigma_\epsilon^2$$

Variance grows linearly; PI width grows as $\sqrt{h}$.

### Gaussian Prediction Intervals

Under normality:
$$y_{T+h}|y_{1:T} \sim N(\hat{y}_{T+h|T}, \sigma_h^2)$$

95% PI: $\hat{y}_{T+h|T} \pm 1.96\sigma_h$
80% PI: $\hat{y}_{T+h|T} \pm 1.28\sigma_h$

### Accounting for Parameter Uncertainty

When parameters are estimated, additional uncertainty:
$$\text{Var}(e_{T+h|T}) \approx \sigma_h^2 + \frac{\sigma_h^2}{n}\sum_{j=0}^{h-1}\left(\frac{\partial\psi_j}{\partial\theta}\right)^2\text{Var}(\hat{\theta})$$

For large samples, parameter uncertainty is small relative to intrinsic forecast uncertainty.

## Algorithm/Model Sketch

**Computing Prediction Intervals:**

```
1. Fit model, estimate parameters θ̂ and σ̂²
2. For each horizon h = 1, ..., H:
   a. Compute point forecast ŷ_{T+h|T}
   b. Calculate ψ₀, ψ₁, ..., ψ_{h-1} (MA coefficients)
   c. Compute σ̂ₕ² = σ̂² × Σψⱼ²
   d. Form interval: ŷ_{T+h|T} ± z_{α/2} × σ̂ₕ

3. For non-normal data, use bootstrap:
   a. Generate B bootstrap samples
   b. Refit model on each
   c. Generate forecasts
   d. Take percentiles of forecast distribution
```

**Bootstrap Prediction Intervals:**
```
For b = 1 to B:
   1. Sample residuals with replacement: ε*[1:n]
   2. Generate bootstrap series y* using model
   3. Refit model to y*
   4. Generate forecasts ŷ*_{T+1:T+H}

Take 2.5% and 97.5% percentiles → 95% PI
```

## Common Pitfalls

1. **Ignoring PI widening**: For I(d) processes, PIs grow without bound. Don't expect tight long-range forecasts.

2. **Assuming constant width**: Only stationary AR(∞) processes have bounded PI. Most models have widening intervals.

3. **Undercoverage**: If actual coverage < nominal, model may be misspecified or variance underestimated.

4. **Overcoverage**: If actual coverage >> nominal, model may be over-conservative or wrong distributional assumption.

5. **Non-normality**: For skewed or heavy-tailed data, Gaussian PIs may be too narrow. Use bootstrap.

6. **Ignoring parameter uncertainty**: Small sample → parameter estimates uncertain → PIs wider than formula suggests.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Generate AR(1) data
np.random.seed(42)
phi = 0.7
sigma = 1.0
n = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + np.random.randn() * sigma

# Fit model
model = ARIMA(y, order=(1, 0, 0)).fit()

# Get forecasts with prediction intervals
forecast_obj = model.get_forecast(steps=20)
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% PI

print("Forecast with 95% PI:")
for h in [1, 5, 10, 20]:
    print(f"  h={h}: {forecast.iloc[h-1]:.2f} "
          f"[{conf_int.iloc[h-1, 0]:.2f}, {conf_int.iloc[h-1, 1]:.2f}]")

# Theoretical width for AR(1)
phi_hat = model.arparams[0]
sigma_hat = np.sqrt(model.scale)
for h in [1, 5, 10, 20]:
    var_h = sigma_hat**2 * (1 - phi_hat**(2*h)) / (1 - phi_hat**2)
    width_theory = 2 * 1.96 * np.sqrt(var_h)
    width_actual = conf_int.iloc[h-1, 1] - conf_int.iloc[h-1, 0]
    print(f"h={h}: Theory width={width_theory:.2f}, Actual={width_actual:.2f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why do prediction intervals widen with forecast horizon for most time series models?</summary>

<div class="answer">
<strong>Answer:</strong> Future shocks are unknown and accumulate over time.

For h-step forecast, we don't know $\epsilon_{T+1}, \ldots, \epsilon_{T+h}$. The forecast error:
$$e_{T+h|T} = \sum_{j=0}^{h-1}\psi_j\epsilon_{T+h-j}$$

More unknown shocks → more variance → wider interval.

**Exceptions:**
- Mean-reverting processes (stationary AR) converge to unconditional variance
- But for random walk (unit root), variance grows linearly with h

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting tight long-range forecasts. For I(1) processes, 1-year-ahead PI is much wider than 1-day-ahead. This is fundamental, not a modeling failure.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What does it mean if your prediction intervals have 85% coverage when they're supposed to have 95%?</summary>

<div class="answer">
<strong>Answer:</strong> **Undercoverage** — actual values fall outside the PI more often than expected.

**Possible causes:**
1. **Model misspecification**: True process not captured (missing components)
2. **Non-normality**: Heavy tails cause more extreme values
3. **Variance underestimated**: $\hat{\sigma}$ too small
4. **Structural changes**: Model fit to stable period, tested on volatile period
5. **Parameter uncertainty ignored**: Especially problematic in small samples

**Remedies:**
- Use bootstrap PIs
- Check residual diagnostics
- Consider heavier-tailed distributions
- Widen intervals manually (e.g., use 99% for conservative 95%)

<div class="pitfall">
<strong>Common pitfall:</strong> Trusting nominal coverage without validation. Always check empirical coverage on holdout data.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the h-step forecast variance for ARIMA(0,1,0) (random walk).</summary>

<div class="answer">
<strong>Answer:</strong> $\sigma_h^2 = h\sigma_\epsilon^2$

**Derivation:**
Random walk: $y_t = y_{t-1} + \epsilon_t$

The h-step forecast:
$$y_{T+h} = y_T + \sum_{j=1}^{h}\epsilon_{T+j}$$

Best forecast: $\hat{y}_{T+h|T} = y_T$ (current value)

Forecast error:
$$e_{T+h|T} = y_{T+h} - y_T = \sum_{j=1}^{h}\epsilon_{T+j}$$

Since $\epsilon_j$ are independent:
$$\text{Var}(e_{T+h|T}) = \sum_{j=1}^{h}\sigma_\epsilon^2 = h\sigma_\epsilon^2$$

**PI:** $y_T \pm 1.96\sigma_\epsilon\sqrt{h}$

Width grows as $\sqrt{h}$.

<div class="pitfall">
<strong>Common pitfall:</strong> For random walk, long-horizon PIs become very wide. A 100-step-ahead 95% PI is 10× wider than 1-step-ahead.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why does a stationary AR(1) have bounded prediction interval width as $h \to \infty$?</summary>

<div class="answer">
<strong>Answer:</strong> For stationary AR(1) with $|\phi| < 1$:

$$\sigma_h^2 = \sigma_\epsilon^2\frac{1-\phi^{2h}}{1-\phi^2}$$

As $h \to \infty$: $\phi^{2h} \to 0$

$$\lim_{h\to\infty}\sigma_h^2 = \frac{\sigma_\epsilon^2}{1-\phi^2} = \text{Var}(y_t)$$

**Intuition:** For stationary processes, distant future values are independent of current observation. The forecast converges to unconditional mean, and uncertainty converges to unconditional variance.

The PI width converges to $2 \times 1.96 \times \sqrt{\text{Var}(y_t)}$ — the interval you'd give without any data.

<div class="pitfall">
<strong>Common pitfall:</strong> Bounded PI doesn't mean narrow PI. The unconditional variance can still be large, especially for $\phi$ near 1.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> Your model produces 95% prediction intervals, but stakeholders want to know the "worst case." How do you translate PIs for business use?</summary>

<div class="answer">
<strong>Answer:</strong> Several approaches:

1. **Use higher confidence level**: 99% PI gives more conservative bounds
   - Upper 99% PI ≈ upper bound for planning

2. **Report specific percentiles**:
   - "95% chance demand is below X"
   - "5% chance it exceeds Y"

3. **Scenario analysis**:
   - Best case: lower 80% bound
   - Expected: point forecast
   - Worst case: upper 95% or 99% bound

4. **Distribution summary**:
   - Most likely range: 50% PI
   - Reasonable range: 80% PI
   - Extreme scenarios: 95% PI

5. **Risk quantiles**: "10% chance of loss exceeding $Z"

**Key message:** PIs are probability statements. "Worst case" depends on acceptable risk level.

<div class="pitfall">
<strong>Common pitfall:</strong> Using upper 95% bound as "worst case" — this still has 2.5% chance of being exceeded. For true tail risk, use higher percentiles or extreme value methods.
</div>
</div>
</details>

## References

1. Chatfield, C. (1993). Calculating interval forecasts. *Journal of Business & Economic Statistics*, 11(2), 121-135.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 5.
3. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 5.
4. Thombs, L. A., & Schucany, W. R. (1990). Bootstrap prediction intervals for autoregression. *JASA*, 85(410), 486-492.
