# ETS Framework

<div class="interview-summary">
<strong>Interview Summary:</strong> ETS (Error-Trend-Seasonal) is a state space framework unifying exponential smoothing methods. Named by three components: Error (A/M), Trend (N/A/Ad/M/Md), Seasonal (N/A/M). Provides likelihood-based estimation and proper prediction intervals. ETS(A,A,A) = additive Holt-Winters. Total of 30 model variants.
</div>

## Core Definitions

**ETS Taxonomy:**

- **E (Error)**: Additive (A) or Multiplicative (M)
- **T (Trend)**: None (N), Additive (A), Additive damped (Ad), Multiplicative (M), Multiplicative damped (Md)
- **S (Seasonal)**: None (N), Additive (A), Multiplicative (M)

**Notation:** ETS(E,T,S)

**Examples:**
- ETS(A,N,N) = Simple Exponential Smoothing
- ETS(A,A,N) = Holt's linear
- ETS(A,Ad,N) = Damped trend
- ETS(A,A,A) = Additive Holt-Winters
- ETS(M,A,M) = Multiplicative error, additive trend, multiplicative seasonal

## Math and Derivations

### State Space Form

**General form:**
$$y_t = w(\mathbf{x}_{t-1}) + r(\mathbf{x}_{t-1})\epsilon_t$$
$$\mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1})\epsilon_t$$

where $\mathbf{x}_t$ is the state vector (level, trend, seasonal components).

### ETS(A,A,N): Additive Error, Additive Trend, No Seasonal

Measurement: $y_t = \ell_{t-1} + b_{t-1} + \epsilon_t$

State transitions:
$$\ell_t = \ell_{t-1} + b_{t-1} + \alpha\epsilon_t$$
$$b_t = b_{t-1} + \beta\epsilon_t$$

Matrix form:
$$\begin{pmatrix} \ell_t \\ b_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} \ell_{t-1} \\ b_{t-1} \end{pmatrix} + \begin{pmatrix} \alpha \\ \beta \end{pmatrix}\epsilon_t$$

### ETS(M,A,M): Multiplicative Error and Seasonal

Measurement: $y_t = (\ell_{t-1} + b_{t-1})s_{t-m}(1 + \epsilon_t)$

This means: $\epsilon_t = \frac{y_t - (\ell_{t-1} + b_{t-1})s_{t-m}}{(\ell_{t-1} + b_{t-1})s_{t-m}}$

State transitions:
$$\ell_t = (\ell_{t-1} + b_{t-1})(1 + \alpha\epsilon_t)$$
$$b_t = b_{t-1} + \beta(\ell_{t-1} + b_{t-1})\epsilon_t$$
$$s_t = s_{t-m}(1 + \gamma\epsilon_t)$$

### Likelihood Function

For additive errors:
$$L(\boldsymbol{\theta}|\mathbf{y}) = \prod_{t=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\epsilon_t^2}{2\sigma^2}\right)$$

For multiplicative errors:
$$L(\boldsymbol{\theta}|\mathbf{y}) = \prod_{t=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}\mu_t}\exp\left(-\frac{\epsilon_t^2}{2\sigma^2}\right)$$

where $\mu_t$ is the one-step-ahead forecast.

### Prediction Intervals

State space formulation enables analytical or simulation-based prediction intervals:

**Analytical** (for some models):
$$\text{Var}(y_{T+h}|y_{1:T}) = \sigma^2 \sum_{j=0}^{h-1}c_j^2$$

**Simulation** (general):
1. Sample future errors $\epsilon_{T+1}, \ldots, \epsilon_{T+h}$
2. Generate sample paths using state equations
3. Compute percentiles of forecast distribution

## Algorithm/Model Sketch

**Model Selection with ETS:**

```
1. Consider all valid ETS combinations:
   - 30 models total (some multiplicative error combinations unstable)
   - Stable models: about 15-20 depending on data

2. For each model:
   - Estimate parameters by MLE
   - Compute AIC/BIC

3. Select model with lowest AIC (or BIC)

4. Validate:
   - Check residual diagnostics
   - Compare forecast accuracy on holdout

5. Generate forecasts with prediction intervals
```

**Valid/Stable Models:**
- Multiplicative error requires positive data
- Some combinations are unstable (e.g., M,Md,M with certain parameters)
- ETS implementations typically restrict to admissible parameter space

## Common Pitfalls

1. **Assuming ETS = Holt-Winters**: ETS is broader—includes multiplicative error variants and provides proper statistical framework.

2. **Ignoring multiplicative error**: For positive data with variance proportional to level, multiplicative error often fits better.

3. **Model averaging**: Instead of selecting one model, averaging forecasts from multiple ETS models can improve accuracy.

4. **Large seasonal period**: ETS with $m > 24$ is often impractical. Use Fourier terms or TBATS instead.

5. **Negative forecasts**: Additive models can forecast negatives. For positive data, prefer multiplicative components.

6. **Prediction interval coverage**: Check that actual coverage matches nominal (e.g., 95% intervals should contain ~95% of observations).

## Mini Example

```python
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Generate data with trend and multiplicative seasonality
np.random.seed(42)
n = 120
t = np.arange(n)
level = 100 + 0.5 * t
seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
y = level * seasonal * (1 + 0.05 * np.random.randn(n))

# Fit various ETS models
models = {
    'ETS(A,A,A)': {'trend': 'add', 'seasonal': 'add'},
    'ETS(A,A,M)': {'trend': 'add', 'seasonal': 'mul'},
    'ETS(A,Ad,A)': {'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
    'ETS(A,Ad,M)': {'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
}

results = {}
for name, params in models.items():
    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=12,
            **params
        ).fit()
        results[name] = {'AIC': model.aic, 'BIC': model.bic}
    except:
        results[name] = {'AIC': np.inf, 'BIC': np.inf}

print("Model Comparison:")
for name, metrics in sorted(results.items(), key=lambda x: x[1]['AIC']):
    print(f"  {name}: AIC={metrics['AIC']:.1f}, BIC={metrics['BIC']:.1f}")

# Best model
best_model = min(results.items(), key=lambda x: x[1]['AIC'])[0]
print(f"\nBest model by AIC: {best_model}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the advantage of the ETS framework over traditional exponential smoothing formulations?</summary>

<div class="answer">
<strong>Answer:</strong> ETS provides:

1. **Statistical foundation**: State space formulation enables proper likelihood inference
2. **Model selection**: AIC/BIC for comparing all 30 variants systematically
3. **Proper prediction intervals**: Based on forecast error distribution, not ad-hoc formulas
4. **Unified framework**: All exponential smoothing methods in one consistent notation
5. **Automatic selection**: Can search over models algorithmically

Traditional formulations gave point forecasts but lacked principled interval estimation and model comparison.

<div class="pitfall">
<strong>Common pitfall:</strong> Using traditional Holt-Winters formulas then computing intervals with ETS assumptions. The interval formulas depend on the error structure — must be consistent.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> When would you prefer multiplicative error (M) over additive error (A)?</summary>

<div class="answer">
<strong>Answer:</strong> Use multiplicative error when:

1. **Variance scales with level**: Higher values have larger absolute errors but similar percentage errors
2. **Positive data only**: Multiplicative error requires $y_t > 0$
3. **Percentage errors are meaningful**: Business contexts where 10% error is similar regardless of level
4. **Heteroskedasticity**: Variance is not constant over time

**Diagnostic:** Plot residuals vs. fitted values. If variance increases with fitted values → multiplicative error.

**Mathematical interpretation:**
- Additive: $y_t = \mu_t + \epsilon_t$ (constant variance)
- Multiplicative: $y_t = \mu_t(1 + \epsilon_t)$ (variance proportional to $\mu_t^2$)

<div class="pitfall">
<strong>Common pitfall:</strong> Using additive error for sales/financial data where percentage errors are natural. This underestimates uncertainty at high values.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Write the state space equations for ETS(A,N,N) — simple exponential smoothing.</summary>

<div class="answer">
<strong>Answer:</strong>

**Measurement equation:**
$$y_t = \ell_{t-1} + \epsilon_t$$

**State transition:**
$$\ell_t = \ell_{t-1} + \alpha\epsilon_t$$

**Or equivalently:**
$$\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$$

This is exactly SES. The state is just the level $\ell_t$. Forecast: $\hat{y}_{t+h|t} = \ell_t$ for all $h$.

**Variance of h-step forecast error:**
$$\text{Var}(y_{t+h} - \hat{y}_{t+h|t}) = \sigma^2[1 + (h-1)\alpha^2]$$

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting that ETS(A,N,N) prediction intervals widen with horizon. Flat forecasts don't mean constant uncertainty.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why are some ETS model combinations inadmissible or unstable?</summary>

<div class="answer">
<strong>Answer:</strong> Certain combinations lead to:

1. **Negative components**: Multiplicative seasonal with additive trend can produce $\ell_t + b_t < 0$, making $y_t = (\ell_t + b_t) \times s_t$ negative even with positive seasonals.

2. **Explosive variance**: Some multiplicative error combinations have variance that grows exponentially with horizon.

3. **Non-identifiability**: Parameter combinations that produce identical forecasts.

**Specifically problematic:**
- ETS(M,M,*) — multiplicative trend with multiplicative error can explode
- ETS(M,*,M) — can give negative forecasts or infinite variance

**Admissible region:** Parameters must satisfy constraints to ensure positive forecasts and bounded variance. Software enforces these.

<div class="pitfall">
<strong>Common pitfall:</strong> Manually setting parameters outside admissible bounds. Always use constrained optimization or let software handle admissibility.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You run automatic ETS selection and get ETS(M,Ad,M) with AIC much lower than alternatives. What checks should you perform before accepting this model?</summary>

<div class="answer">
<strong>Answer:</strong> Before accepting:

1. **Check residuals:**
   - Plot standardized residuals — should look like white noise
   - ACF of residuals — no significant spikes
   - Ljung-Box test — fail to reject white noise

2. **Verify assumptions:**
   - Data is positive (required for multiplicative)
   - Variance scales with level (justifies M error)
   - Seasonal pattern is proportional (justifies M seasonal)

3. **Compare forecasts:**
   - Out-of-sample validation if possible
   - Do forecasts look reasonable?
   - Check prediction interval coverage

4. **Parameter reasonableness:**
   - Damping parameter φ — should be 0.8-0.98
   - α, β, γ — not at boundaries

5. **Compare to simpler models:**
   - If ETS(A,A,M) is close, prefer simpler
   - Log-transform + ETS(A,*,A) might be equivalent

<div class="pitfall">
<strong>Common pitfall:</strong> Accepting complex model without validation. ETS(M,Ad,M) has many parameters — risk of overfitting. Always validate on holdout data.
</div>
</div>
</details>

## References

1. Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *IJF*, 18(3), 439-454.
2. Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 8.
4. De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *JASA*, 106(496), 1513-1527.
