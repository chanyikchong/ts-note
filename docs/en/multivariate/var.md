# Vector Autoregression (VAR)

<div class="interview-summary">
<strong>Interview Summary:</strong> VAR models multiple time series jointly, where each variable depends on its own lags and lags of others. VAR(p): $\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{y}_{t-1} + \cdots + \mathbf{A}_p\mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t$. Useful for forecasting interrelated series and analyzing dynamic relationships. Estimate by OLS equation-by-equation. Select order with AIC/BIC. Check stability via eigenvalues.
</div>

## Core Definitions

**VAR(p) Model:**
$$\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{y}_{t-1} + \mathbf{A}_2\mathbf{y}_{t-2} + \cdots + \mathbf{A}_p\mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t$$

where:
- $\mathbf{y}_t$: k×1 vector of variables
- $\mathbf{c}$: k×1 constant vector
- $\mathbf{A}_i$: k×k coefficient matrices
- $\boldsymbol{\epsilon}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})$: k×1 error vector

**Compact Form:**
$$\mathbf{A}(L)\mathbf{y}_t = \mathbf{c} + \boldsymbol{\epsilon}_t$$

where $\mathbf{A}(L) = \mathbf{I} - \mathbf{A}_1 L - \cdots - \mathbf{A}_p L^p$

**Stationarity Condition:** All eigenvalues of companion matrix inside unit circle.

## Math and Derivations

### Bivariate VAR(1) Example

For variables $(y_{1t}, y_{2t})$:
$$\begin{pmatrix} y_{1t} \\ y_{2t} \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}\begin{pmatrix} y_{1,t-1} \\ y_{2,t-1} \end{pmatrix} + \begin{pmatrix} \epsilon_{1t} \\ \epsilon_{2t} \end{pmatrix}$$

Written out:
$$y_{1t} = c_1 + a_{11}y_{1,t-1} + a_{12}y_{2,t-1} + \epsilon_{1t}$$
$$y_{2t} = c_2 + a_{21}y_{1,t-1} + a_{22}y_{2,t-1} + \epsilon_{2t}$$

Cross-coefficients $a_{12}, a_{21}$ capture dynamic spillovers.

### Companion Form

VAR(p) can be written as VAR(1) in higher dimension:
$$\boldsymbol{\xi}_t = \mathbf{F}\boldsymbol{\xi}_{t-1} + \mathbf{v}_t$$

where $\boldsymbol{\xi}_t = (\mathbf{y}_t', \mathbf{y}_{t-1}', \ldots, \mathbf{y}_{t-p+1}')'$ and $\mathbf{F}$ is the companion matrix.

Stationarity: eigenvalues of $\mathbf{F}$ have modulus < 1.

### MA(∞) Representation

Stationary VAR has moving average form:
$$\mathbf{y}_t = \boldsymbol{\mu} + \sum_{i=0}^{\infty}\boldsymbol{\Phi}_i\boldsymbol{\epsilon}_{t-i}$$

$\boldsymbol{\Phi}_i$ are impulse response matrices: $\boldsymbol{\Phi}_i^{jk}$ = response of variable j to shock in variable k at lag i.

### Forecast Error Variance Decomposition

Variance of h-step forecast error for variable j:
$$\sigma_j^2(h) = \sum_{i=0}^{h-1}\sum_{k=1}^{K}(\Phi_i^{jk})^2\sigma_k^2$$

Contribution of variable k to variance of j at horizon h.

## Algorithm/Model Sketch

**VAR Estimation:**

```
1. Determine optimal lag order p:
   - Fit VAR(1), VAR(2), ..., VAR(p_max)
   - Select p minimizing AIC or BIC

2. Estimate by OLS:
   - Each equation can be estimated separately
   - OLS is consistent and efficient (same regressors)

3. Check stability:
   - Compute eigenvalues of companion matrix
   - All |λᵢ| < 1 for stationarity

4. Diagnostics:
   - Test residuals for autocorrelation (multivariate LB)
   - Test for normality
   - Check for heteroskedasticity

5. Analysis:
   - Impulse responses
   - Forecast error variance decomposition
   - Granger causality tests
```

## Common Pitfalls

1. **Too many parameters**: VAR(p) with k variables has k + k²p parameters. Overfitting is easy with high k or p.

2. **Non-stationary variables**: VAR requires stationarity. Use differences or VECM for I(1) variables.

3. **Structural interpretation**: Reduced-form VAR shows correlations, not causation. Use structural VAR (SVAR) for causal claims.

4. **Ignoring cointegration**: If variables are cointegrated, restricted VECM is more efficient than unrestricted VAR in differences.

5. **Over-interpreting IRFs**: Impulse responses depend on ordering (Cholesky) or identification assumptions.

6. **Forgetting contemporaneous correlation**: $\boldsymbol{\Sigma}$ is not diagonal; shocks are correlated across equations.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.api import VAR

# Generate bivariate VAR(1) data
np.random.seed(42)
n = 200
A = np.array([[0.5, 0.3],
              [0.2, 0.4]])
c = np.array([1, 2])

y = np.zeros((n, 2))
for t in range(1, n):
    y[t] = c + A @ y[t-1] + np.random.randn(2) * 0.5

# Fit VAR
model = VAR(y)

# Select lag order
lag_order = model.select_order(maxlags=8)
print("Lag selection:")
print(lag_order.summary())

# Fit VAR(1)
results = model.fit(1)
print("\nCoefficient matrix A:")
print(results.coefs[0])
print(f"\nTrue A:\n{A}")

# Impulse response
irf = results.irf(10)
print(f"\nIRF: Response of y1 to y2 shock at lag 5: {irf.irfs[5, 0, 1]:.3f}")

# Forecast
forecast = results.forecast(y[-1:], steps=5)
print(f"\n5-step forecast:\n{forecast}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the advantage of VAR over fitting separate univariate models?</summary>

<div class="answer">
<strong>Answer:</strong> VAR captures cross-variable dynamics:

1. **Dynamic interactions**: How $y_1$ affects future $y_2$ and vice versa
2. **Joint forecasting**: Uses information from all variables
3. **Correlated errors**: Accounts for contemporaneous shocks
4. **Policy analysis**: Impulse responses show system-wide effects

**Example:** GDP and inflation. VAR captures:
- Past GDP affecting future inflation (demand effects)
- Past inflation affecting future GDP (real balance effects)
- Correlated supply shocks hitting both

Separate ARIMAs miss these interactions.

<div class="pitfall">
<strong>Common pitfall:</strong> Using VAR when variables are unrelated. With k variables, you estimate k² coefficients per lag—wasteful if many are zero. Consider sparse VAR or variable selection.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Why is the ordering of variables important for impulse response analysis?</summary>

<div class="answer">
<strong>Answer:</strong> Standard IRFs use Cholesky decomposition of $\boldsymbol{\Sigma}$, which depends on variable ordering.

**Cholesky:** $\boldsymbol{\Sigma} = \mathbf{PP}'$ where $\mathbf{P}$ is lower triangular.

This implies:
- First variable's shock is "structural" (not affected by others contemporaneously)
- Later variables respond to earlier ones within same period

**Different orderings → different IRFs**

**Example:** Order (GDP, Inflation) vs (Inflation, GDP)
- First ordering: GDP shock affects inflation immediately
- Second ordering: Inflation shock affects GDP immediately

**Solutions:**
- Use theory to justify ordering
- Use structural VAR with explicit identification
- Report sensitivity to ordering

<div class="pitfall">
<strong>Common pitfall:</strong> Reporting IRFs without stating ordering or justifying identification. Results may be driven by arbitrary ordering choice.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> For VAR(1) with coefficient matrix $\mathbf{A}$, what is the stationarity condition?</summary>

<div class="answer">
<strong>Answer:</strong> All eigenvalues of $\mathbf{A}$ must have modulus less than 1.

**Why:**
VAR(1): $\mathbf{y}_t = \mathbf{c} + \mathbf{A}\mathbf{y}_{t-1} + \boldsymbol{\epsilon}_t$

Iterating backward:
$$\mathbf{y}_t = (\mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \cdots)\mathbf{c} + \sum_{j=0}^{\infty}\mathbf{A}^j\boldsymbol{\epsilon}_{t-j}$$

This converges iff $\mathbf{A}^j \to 0$, which requires all eigenvalues inside unit circle.

**For bivariate:**
If $\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

Eigenvalues: $\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}$

Need $|\lambda_1| < 1$ and $|\lambda_2| < 1$.

<div class="pitfall">
<strong>Common pitfall:</strong> Checking only diagonal elements. Even if $|a|, |d| < 1$, off-diagonal terms can make system unstable.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How many parameters does a VAR(p) model with k variables have?</summary>

<div class="answer">
<strong>Answer:</strong> $k + k^2 p + \frac{k(k+1)}{2}$

**Breakdown:**
- $k$ constant terms (vector $\mathbf{c}$)
- $k^2 \times p$ coefficients (p matrices of size k×k)
- $\frac{k(k+1)}{2}$ variance-covariance parameters (symmetric $\boldsymbol{\Sigma}$)

**Example:** k=3 variables, p=4 lags:
- Constants: 3
- AR coefficients: 9 × 4 = 36
- Covariance: 6
- Total: 45 parameters

**Implications:**
- Parameters grow as $k^2$
- With limited data, overfitting is severe
- Consider restricted VAR, BVAR, or variable selection

<div class="pitfall">
<strong>Common pitfall:</strong> Fitting large VAR with small samples. Rule of thumb: need at least 10-20 observations per parameter.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You fit VAR(2) to 3 quarterly macro variables. The residual autocorrelation test rejects at lag 4. What do you do?</summary>

<div class="answer">
<strong>Answer:</strong> Lag 4 autocorrelation with quarterly data suggests annual seasonality. Options:

1. **Increase lag order**: Try VAR(4) or VAR(5) to capture annual dynamics

2. **Add seasonal dummies**: Include Q1, Q2, Q3 indicators as exogenous variables

3. **Seasonally adjust**: Pre-filter data to remove seasonality

4. **VARX**: Add seasonal Fourier terms as exogenous regressors

**Diagnostic process:**
1. Check if all three residuals show lag-4 pattern
2. Fit VAR(4) and re-test
3. Compare AIC: VAR(2) with seasonals vs VAR(4)
4. Verify residuals now pass tests

**Consideration:** More lags = more parameters. If sample is small, prefer seasonal dummies over VAR(4).

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring seasonal patterns in macro data. Annual effects are common; quarterly VAR should capture them explicitly.
</div>
</div>
</details>

## References

1. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapters 10-11.
3. Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
4. Stock, J. H., & Watson, M. W. (2001). Vector autoregressions. *Journal of Economic Perspectives*, 15(4), 101-115.
