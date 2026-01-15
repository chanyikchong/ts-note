# Granger Causality

<div class="interview-summary">
<strong>Interview Summary:</strong> Granger causality tests whether past values of X help predict Y beyond Y's own history. It's about predictive precedence, not true causation. Test via F-test on VAR coefficients. X Granger-causes Y if coefficients on lagged X in Y's equation are jointly significant. Bidirectional causality is possible. Sensitive to omitted variables and lag selection.
</div>

## Core Definitions

**Granger Causality:** X Granger-causes Y if:
$$E[Y_t | Y_{t-1}, Y_{t-2}, \ldots, X_{t-1}, X_{t-2}, \ldots] \neq E[Y_t | Y_{t-1}, Y_{t-2}, \ldots]$$

Past X provides predictive information about Y beyond Y's own past.

**Non-causality:** X does NOT Granger-cause Y if knowing past X doesn't improve prediction of Y.

**Bivariate VAR Test:**
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + \sum_{i=1}^{p}\beta_i x_{t-i} + \epsilon_t$$

$H_0$: $\beta_1 = \beta_2 = \cdots = \beta_p = 0$ (X does not Granger-cause Y)

## Math and Derivations

### F-Test for Granger Causality

**Restricted model:** AR(p) for Y only
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + u_t$$

**Unrestricted model:** VAR including X
$$y_t = c + \sum_{i=1}^{p}\alpha_i y_{t-i} + \sum_{i=1}^{p}\beta_i x_{t-i} + \epsilon_t$$

**F-statistic:**
$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T-2p-1)}$$

Under $H_0$: $F \sim F_{p, T-2p-1}$

### Wald Test (for VAR)

In VAR framework, test:
$$H_0: \mathbf{R}\boldsymbol{\beta} = \mathbf{0}$$

where $\mathbf{R}$ selects the coefficients on lagged X in Y's equation.

Wald statistic: $W = (\mathbf{R}\hat{\boldsymbol{\beta}})'[\mathbf{R}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{R}'\hat{\sigma}^2]^{-1}(\mathbf{R}\hat{\boldsymbol{\beta}})$

Under $H_0$: $W \sim \chi^2_p$

### Instantaneous Causality

Tests whether current X helps predict current Y (beyond lagged effects):

$$\text{Cov}(\epsilon_{yt}, \epsilon_{xt}) \neq 0$$

This tests contemporaneous correlation, not temporal precedence.

### Block Exogeneity

In multivariate system, test whether a group of variables Granger-causes another group.

Joint test on all relevant coefficient matrices.

## Algorithm/Model Sketch

**Granger Causality Test Procedure:**

```
1. Determine if series are stationary
   - If I(1), difference or use Toda-Yamamoto approach
   - Standard GC tests require stationarity

2. Select optimal lag order
   - Use AIC/BIC on bivariate VAR
   - Or use same p for all tests (consistency)

3. Estimate unrestricted VAR(p)

4. Perform Wald/F test:
   - H0: Coefficients on lagged X = 0 (in Y equation)
   - Reject → X Granger-causes Y

5. Test reverse direction:
   - H0: Coefficients on lagged Y = 0 (in X equation)
   - Reject → Y Granger-causes X

6. Interpret:
   - Both reject: bidirectional causality
   - One rejects: unidirectional causality
   - Neither rejects: no Granger causality
```

**Toda-Yamamoto Approach (for I(1) series):**
1. Determine maximum integration order d_max
2. Fit VAR(p + d_max)
3. Test only coefficients on first p lags
4. Avoids issues with pretesting for unit roots

## Common Pitfalls

1. **Confusing with true causation**: Granger causality is predictive precedence, not causal mechanism. Correlation can arise from common causes.

2. **Omitted variable bias**: If Z causes both X and Y with different lags, you may find spurious GC between X and Y.

3. **Wrong lag selection**: Too few lags → miss true effects. Too many → lose power and introduce noise.

4. **Non-stationary data**: Standard F-tests have wrong distribution with unit roots. Use augmented lag approach or error correction.

5. **Multiple testing**: Testing many pairs inflates Type I error. Adjust significance level.

6. **Contemporaneous effects only**: If X and Y move together within a period but not across periods, GC won't detect it.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

# Generate data where X Granger-causes Y but not vice versa
np.random.seed(42)
n = 200

# X is independent AR(1)
x = np.zeros(n)
for t in range(1, n):
    x[t] = 0.7 * x[t-1] + np.random.randn()

# Y depends on own lag and lagged X
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + np.random.randn()

# Stack data
data = np.column_stack([y, x])

# Granger causality tests
print("=== Does X Granger-cause Y? ===")
gc_x_to_y = grangercausalitytests(data, maxlag=4, verbose=True)

print("\n=== Does Y Granger-cause X? ===")
gc_y_to_x = grangercausalitytests(data[:, ::-1], maxlag=4, verbose=True)

# Using VAR
model = VAR(data)
results = model.fit(2)
print("\n=== VAR-based Granger Causality ===")
print(results.test_causality('y1', 'y2', kind='f'))  # X → Y
print(results.test_causality('y2', 'y1', kind='f'))  # Y → X
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why is Granger causality not the same as true causation?</summary>

<div class="answer">
<strong>Answer:</strong> Granger causality only measures predictive precedence—whether past X helps predict Y. It doesn't establish causal mechanism.

**Why they differ:**
1. **Omitted variables**: Z may cause both X and Y with different lags, creating spurious GC
2. **Common causes**: X and Y may both respond to unobserved factor
3. **Spurious correlation**: Can find GC even in independent series by chance
4. **Measurement timing**: If X and Y are measured at different times, GC reflects measurement, not causation

**Example:** Ice cream sales Granger-cause drownings (both caused by summer heat with different lags).

<div class="pitfall">
<strong>Common pitfall:</strong> Claiming X causes Y based on GC test. Always say "X Granger-causes Y" or "X has predictive power for Y"—not "X causes Y."
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What does bidirectional Granger causality mean? Is this common?</summary>

<div class="answer">
<strong>Answer:</strong> Bidirectional GC means both X → Y and Y → X (each helps predict the other). This is common in economics.

**Examples:**
- GDP ↔ Employment (economic activity and labor market interact)
- Prices ↔ Wages (wage-price spiral)
- Interest rates ↔ Exchange rates (monetary policy and currency markets)

**Interpretation:**
- Feedback relationship
- Both variables contain unique predictive information
- System is interdependent

**Caution:** Bidirectional GC doesn't mean simultaneous causation—it means mutual predictive value across time.

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting one-way causality. In complex systems, feedback is the rule rather than exception.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> What are the degrees of freedom for the Granger causality F-test with p lags and T observations?</summary>

<div class="answer">
<strong>Answer:</strong> $F_{p, T-2p-1}$ (numerator df = p, denominator df = T - 2p - 1)

**Derivation:**
- Restricted model: p parameters (p lags of Y + constant)
- Unrestricted model: 2p + 1 parameters (p lags of Y + p lags of X + constant)
- Restriction: p parameters set to zero
- Observations used: T - p (lose p for lags)

Numerator df = number of restrictions = p
Denominator df = T - p - (2p + 1) = T - 3p - 1

(Some formulations differ slightly depending on whether constant is counted.)

**Practical:** Use software; these details are handled automatically.

<div class="pitfall">
<strong>Common pitfall:</strong> With short series and many lags, degrees of freedom are low, reducing test power. Balance p against sample size.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How does the Toda-Yamamoto approach handle non-stationary series in Granger causality testing?</summary>

<div class="answer">
<strong>Answer:</strong> Toda-Yamamoto (1995) avoids pretesting for unit roots:

1. Determine maximum integration order $d_{max}$ (usually 1 or 2)
2. Fit VAR(p + $d_{max}$) in levels (don't difference)
3. Test Granger causality on first p lags only
4. Extra $d_{max}$ lags absorb non-stationarity

**Why it works:**
- VAR in levels with extra lags has standard asymptotic distribution for Wald test
- No need to pretest for cointegration
- Robust to I(1) or I(0) series

**Test:**
$$H_0: \beta_1 = \cdots = \beta_p = 0$$

(Ignore $\beta_{p+1}, \ldots, \beta_{p+d_{max}}$)

<div class="pitfall">
<strong>Common pitfall:</strong> Testing all coefficients including extra lags. Only test first p; the extra $d_{max}$ are "nuisance" parameters.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You test Granger causality between oil prices and stock returns using lags 1-8. Results vary: significant at lags 2, 4, 5 but not others. How do you interpret?</summary>

<div class="answer">
<strong>Answer:</strong> This pattern suggests:

1. **Lag selection matters**: Results are sensitive to specification
2. **Possible weak relationship**: Significance appears at some lags by chance
3. **Multiple testing**: Testing 8 specifications inflates false positives

**Recommended approach:**
1. Select lag order FIRST using information criteria (not GC results)
2. Report single test at optimal lag
3. If sensitivity analysis needed, report all results and acknowledge instability
4. Consider Bonferroni correction for multiple tests
5. Validate on out-of-sample data

**If results are inconsistent:**
- Weak evidence for Granger causality
- Relationship may be nonlinear or time-varying
- Consider threshold VAR or regime-switching model

<div class="pitfall">
<strong>Common pitfall:</strong> Selecting lag that gives desired result ("lag shopping"). This is p-hacking; report pre-specified lag or all results.
</div>
</div>
</details>

## References

1. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.
2. Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66(1-2), 225-250.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 11.
4. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 2.
