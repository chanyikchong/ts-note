# Information Criteria

<div class="interview-summary">
<strong>Interview Summary:</strong> Information criteria balance model fit against complexity. AIC = -2log(L) + 2k favors prediction; BIC = -2log(L) + k·log(n) favors true model recovery. Lower is better. AIC tends to select larger models; BIC is more parsimonious. For small samples, use AICc. When criteria disagree, consider your goal: prediction (AIC) vs. inference (BIC).
</div>

## Core Definitions

**AIC (Akaike Information Criterion):**
$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

**BIC (Bayesian/Schwarz Information Criterion):**
$$\text{BIC} = -2\ln(\hat{L}) + k\ln(n)$$

**AICc (Corrected AIC):**
$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

**Components:**
- $\hat{L}$: Maximum likelihood
- $k$: Number of estimated parameters
- $n$: Sample size

## Math and Derivations

### AIC Derivation (Intuition)

AIC minimizes expected Kullback-Leibler divergence between true and fitted model:
$$\text{KL}(f||g_{\hat{\theta}}) = E_f[\ln f(y)] - E_f[\ln g_{\hat{\theta}}(y)]$$

Akaike showed:
$$E[-2\ln g_{\hat{\theta}}(y_{new})] \approx -2\ln g_{\hat{\theta}}(y) + 2k$$

Minimizing AIC approximately minimizes out-of-sample prediction error.

### BIC Derivation (Intuition)

BIC approximates log marginal likelihood:
$$\ln p(y|M) \approx \ln p(y|\hat{\theta},M) - \frac{k}{2}\ln(n)$$

BIC is consistent: if true model is among candidates, BIC selects it with probability → 1 as n → ∞.

### For Gaussian Time Series

With residual variance $\hat{\sigma}^2$:
$$\text{AIC} = n\ln(\hat{\sigma}^2) + 2k$$
$$\text{BIC} = n\ln(\hat{\sigma}^2) + k\ln(n)$$

### Penalty Comparison

| n | AIC penalty | BIC penalty |
|---|-------------|-------------|
| 8 | 2k | 2.08k |
| 20 | 2k | 3.00k |
| 100 | 2k | 4.61k |
| 1000 | 2k | 6.91k |

BIC penalty grows with n; AIC stays constant.

## Algorithm/Model Sketch

**Model Selection Procedure:**

```
1. Define candidate models: M₁, M₂, ..., Mₘ
2. For each model Mᵢ:
   - Fit model by MLE
   - Compute AIC and BIC

3. Rank by criterion:
   - For prediction: prefer AIC (or AICc for small n)
   - For inference: prefer BIC

4. Compare top candidates:
   - ΔAIC < 2: essentially equivalent
   - ΔAIC 2-7: some support for better model
   - ΔAIC > 10: strong support for better model

5. Validate:
   - Check residual diagnostics for selected model
   - Consider out-of-sample testing
```

**Akaike Weights:**
$$w_i = \frac{\exp(-\frac{1}{2}\Delta\text{AIC}_i)}{\sum_j\exp(-\frac{1}{2}\Delta\text{AIC}_j)}$$

Gives probability-like weights for model averaging.

## Common Pitfalls

1. **Treating criteria as absolute**: Only relative values matter. AIC = 1000 vs AIC = 1002 is a meaningful comparison.

2. **Ignoring sample size for AIC/BIC choice**: For n < 40, AICc is essential. For large n, BIC may be too parsimonious.

3. **Using wrong likelihood**: Comparing models with different transformations (log vs. level) requires adjusting likelihood.

4. **Overfitting with AIC in large samples**: As n grows, AIC allows increasingly complex models. Consider BIC for parsimony.

5. **Ignoring ties**: If ΔAIC < 2, models are equivalent. Don't over-interpret small differences.

6. **Forgetting model checking**: Lowest IC doesn't guarantee good model. Always check residuals.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Generate ARMA(1,1) data
np.random.seed(42)
n = 200
phi, theta = 0.7, 0.3
eps = np.random.randn(n + 1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

# Fit candidate models
candidates = [
    ('AR(1)', (1, 0, 0)),
    ('AR(2)', (2, 0, 0)),
    ('MA(1)', (0, 0, 1)),
    ('MA(2)', (0, 0, 2)),
    ('ARMA(1,1)', (1, 0, 1)),
    ('ARMA(2,1)', (2, 0, 1)),
]

results = []
for name, order in candidates:
    model = ARIMA(y, order=order).fit()
    results.append({
        'Model': name,
        'AIC': model.aic,
        'BIC': model.bic,
        'k': sum(order) + 1  # +1 for variance
    })

# Display sorted by AIC
import pandas as pd
df = pd.DataFrame(results).sort_values('AIC')
df['ΔAIC'] = df['AIC'] - df['AIC'].min()
df['ΔBIC'] = df['BIC'] - df['BIC'].min()
print(df.to_string(index=False))

# True model ARMA(1,1) should rank well
print(f"\nBest by AIC: {df.iloc[0]['Model']}")
print(f"Best by BIC: {df.sort_values('BIC').iloc[0]['Model']}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why does BIC tend to select simpler models than AIC?</summary>

<div class="answer">
<strong>Answer:</strong> BIC has a stronger penalty for parameters: $k\ln(n)$ vs $2k$.

For $n > 8$: $\ln(n) > 2$, so BIC penalizes each parameter more.

**Mathematical comparison:**
- AIC adds $2k$ regardless of sample size
- BIC adds $k\ln(n)$, which grows with n

For n = 100: BIC adds 4.6k vs AIC's 2k per parameter.

**Consequence:** BIC requires stronger likelihood improvement to justify additional parameters, leading to simpler models.

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking simpler is always better. BIC can underfit when true model is complex. For forecasting, AIC often wins because it allows capturing more signal.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> When should you use AICc instead of AIC?</summary>

<div class="answer">
<strong>Answer:</strong> Use AICc when sample size is small relative to number of parameters.

**Rule of thumb:** Use AICc when $n/k < 40$.

**Why AICc?**
AIC is derived asymptotically. For small samples, it underpenalizes complexity, leading to overfitting.

AICc correction: $\frac{2k(k+1)}{n-k-1}$

This additional term is large when n ≈ k but vanishes as n → ∞.

**Example:**
- n = 50, k = 5
- AIC penalty: 10
- AICc penalty: 10 + 2(5)(6)/(50-6) ≈ 10 + 1.4 = 11.4

<div class="pitfall">
<strong>Common pitfall:</strong> Using AIC by default without checking n/k ratio. For small samples, AIC systematically selects overly complex models.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the formula for AICc from AIC.</summary>

<div class="answer">
<strong>Answer:</strong> AICc adds a bias correction term:

$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

**Derivation sketch:**
For regression with Gaussian errors, Hurvich and Tsai (1989) showed:

$$E[\text{AIC}] = E[-2\ln L] + 2k$$

has bias when n is small. The exact expected value:

$$E[-2\ln L(\hat{\theta})] + \frac{2kn}{n-k-1}$$

leads to:
$$\text{AICc} = -2\ln L + \frac{2kn}{n-k-1} = \text{AIC} + \frac{2k^2 + 2k}{n-k-1}$$

As $n \to \infty$: $\frac{2k(k+1)}{n-k-1} \to 0$, so AICc → AIC.

<div class="pitfall">
<strong>Common pitfall:</strong> AICc formula assumes residual variance is estimated. For restricted cases, different corrections apply.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Show that BIC is consistent (selects true model) while AIC is not.</summary>

<div class="answer">
<strong>Answer:</strong>

**BIC Consistency:**
For nested models, consider true model $M_0$ (k₀ params) vs larger $M_1$ (k₁ > k₀).

$$\text{BIC}_1 - \text{BIC}_0 = -2(\ln L_1 - \ln L_0) + (k_1 - k_0)\ln n$$

By likelihood ratio theory: $-2(\ln L_1 - \ln L_0) = O_p(1)$ (bounded)
But penalty: $(k_1 - k_0)\ln n \to \infty$

So $P(\text{BIC}_1 > \text{BIC}_0) \to 1$.

**AIC Inconsistency:**
$$\text{AIC}_1 - \text{AIC}_0 = -2(\ln L_1 - \ln L_0) + 2(k_1 - k_0)$$

Extra parameters add fixed penalty 2(k₁-k₀), while likelihood improvement is $O_p(1)$. There's always positive probability that extra parameters improve fit enough to offset the fixed penalty.

<div class="pitfall">
<strong>Common pitfall:</strong> Consistency ≠ better forecasts. AIC minimizes prediction error; BIC identifies true model. Different goals.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> AIC selects ARIMA(2,1,2) while BIC selects ARIMA(1,1,1). AIC difference is 4. Which do you choose?</summary>

<div class="answer">
<strong>Answer:</strong> It depends on your goal and context:

**For forecasting:** Lean toward ARIMA(2,1,2) (AIC's choice)
- ΔAIC = 4 suggests meaningful predictive improvement
- Extra complexity may capture real dynamics

**For interpretation:** Lean toward ARIMA(1,1,1) (BIC's choice)
- Simpler, more interpretable
- Less risk of overfitting

**Recommended approach:**
1. Compare out-of-sample forecast accuracy
2. Check residual diagnostics for both
3. If similar performance, prefer simpler
4. Consider ensemble/averaging

**Decision matrix:**

| Factor | Favors (2,1,2) | Favors (1,1,1) |
|--------|----------------|----------------|
| Large sample | ✓ | |
| Short forecast horizon | ✓ | |
| Complex dynamics expected | ✓ | |
| Interpretability needed | | ✓ |
| Small sample | | ✓ |
| Long forecast horizon | | ✓ |

<div class="pitfall">
<strong>Common pitfall:</strong> Picking one criterion dogmatically. Use domain knowledge, validation, and judgment alongside IC.
</div>
</div>
</details>

## References

1. Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.
2. Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
3. Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference*. Springer.
4. Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297-307.
