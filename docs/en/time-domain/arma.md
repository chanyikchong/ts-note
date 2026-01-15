# ARMA Models

<div class="interview-summary">
<strong>Interview Summary:</strong> ARMA(p,q) combines AR and MA components: $\Phi(L)X_t = \Theta(L)\epsilon_t$. Both ACF and PACF tail off (decay). Stationarity depends on AR part; invertibility depends on MA part. Estimation via MLE. More parsimonious than pure AR or MA when both patterns present.
</div>

## Core Definitions

**ARMA(p,q) Model**:
$$X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + \cdots + \theta_q\epsilon_{t-q}$$

**Lag Operator Form**:
$$\Phi(L)X_t = c + \Theta(L)\epsilon_t$$

where:
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ (AR polynomial)
- $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$ (MA polynomial)

**Mean**: $\mu = \frac{c}{1 - \phi_1 - \cdots - \phi_p}$

**Stationarity**: Roots of $\Phi(z) = 0$ outside unit circle

**Invertibility**: Roots of $\Theta(z) = 0$ outside unit circle

## Math and Derivations

### ARMA(1,1): $X_t = c + \phi X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$

**Stationarity**: $|\phi| < 1$

**Invertibility**: $|\theta| < 1$

**Mean**: $\mu = \frac{c}{1-\phi}$

**Variance** (assuming $\mu = 0$ for simplicity):
$$\gamma(0) = \phi\gamma(1) + \sigma^2(1 + \theta\phi + \theta^2)$$

Solving with $\gamma(1) = \phi\gamma(0) + \theta\sigma^2$:
$$\gamma(0) = \sigma^2 \frac{1 + 2\theta\phi + \theta^2}{1-\phi^2}$$

**ACF**:
$$\rho(1) = \frac{(1+\theta\phi)(\phi+\theta)}{1 + 2\theta\phi + \theta^2}$$
$$\rho(h) = \phi\rho(h-1) \text{ for } h \geq 2$$

Note: ACF decays like AR(1) after lag 1, but $\rho(1)$ differs from AR(1).

### General ARMA(p,q) ACF

For $h > q$:
$$\gamma(h) = \phi_1\gamma(h-1) + \phi_2\gamma(h-2) + \cdots + \phi_p\gamma(h-p)$$

The ACF satisfies the same recursion as AR(p) for lags beyond q. Initial values $\gamma(0), \ldots, \gamma(q)$ depend on both AR and MA parameters.

### Causal and Invertible Representations

**Causal (MA(∞)) form**: If stationary:
$$X_t = \mu + \sum_{j=0}^{\infty}\psi_j\epsilon_{t-j}$$

where $\psi_j$ coefficients come from $\Psi(L) = \Theta(L)/\Phi(L)$.

**Invertible (AR(∞)) form**: If invertible:
$$\Pi(L)(X_t - \mu) = \epsilon_t$$

where $\Pi(L) = \Phi(L)/\Theta(L)$.

### Parameter Redundancy

**Critical**: Ensure AR and MA polynomials share no common roots (factors).

Example: $X_t = 0.5X_{t-1} + \epsilon_t - 0.5\epsilon_{t-1}$

Here $(1-0.5L)X_t = (1-0.5L)\epsilon_t$, which simplifies to $X_t = \epsilon_t$ (white noise!).

This is called a **common factor** or **parameter redundancy**.

## Algorithm/Model Sketch

**Identification:**
```
1. Check stationarity; difference if needed
2. Examine ACF and PACF:
   - Both tail off → ARMA (not pure AR or MA)
   - ACF cuts off → likely MA
   - PACF cuts off → likely AR
3. Use EACF (Extended ACF) or information criteria
4. Fit candidate models
5. Compare AIC/BIC, check residuals
```

**Extended ACF (EACF) Method:**

EACF simplifies identification by iteratively removing AR structure. The resulting table shows "O" pattern indicating (p,q) order.

**Estimation:**

1. **Conditional MLE**: Condition on initial values, maximize likelihood
2. **Exact MLE**: Properly accounts for initial conditions
3. **CSS (Conditional Sum of Squares)**: Minimize squared residuals

Most software uses exact MLE by default.

## Common Pitfalls

1. **Over-parameterization**: ARMA(2,2) often not better than ARMA(1,1). Parsimony matters for forecasting.

2. **Common factor problem**: ARMA(p,q) may reduce to ARMA(p-1,q-1) if polynomials share a root. Check for parameter redundancy.

3. **Local optima**: ARMA likelihood can have multiple modes. Try different starting values.

4. **Near-cancellation**: Parameters close to canceling (e.g., $\phi \approx \theta$) cause estimation instability and inflated standard errors.

5. **Identification confusion**: Both ACF and PACF tail off, but the patterns differ. Focus on overall decay rate and compare with theoretical patterns.

6. **Forgetting conditions**: Need both stationarity (AR roots) AND invertibility (MA roots) outside unit circle.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

# Generate ARMA(1,1) process
np.random.seed(42)
n = 500
phi, theta = 0.7, 0.4
eps = np.random.randn(n + 1)
X = np.zeros(n)

X[0] = eps[1] + theta * eps[0]
for t in range(1, n):
    X[t] = phi * X[t-1] + eps[t+1] + theta * eps[t]

# Both ACF and PACF should tail off
print("ACF (first 6):", np.round(acf(X, nlags=5), 3))
print("PACF (first 6):", np.round(pacf(X, nlags=5), 3))

# Fit ARMA(1,1)
model = ARIMA(X, order=(1, 0, 1)).fit()
print(f"\nTrue: phi={phi}, theta={theta}")
print(f"Estimated: phi={model.arparams[0]:.3f}, theta={model.maparams[0]:.3f}")

# Check for parameter redundancy
print(f"\nPhi - Theta = {abs(model.arparams[0] - model.maparams[0]):.3f}")
# If close to 0, possible near-cancellation

# Compare with pure AR and pure MA
ar_aic = ARIMA(X, order=(2, 0, 0)).fit().aic
ma_aic = ARIMA(X, order=(0, 0, 2)).fit().aic
arma_aic = model.aic
print(f"\nAIC: AR(2)={ar_aic:.1f}, MA(2)={ma_aic:.1f}, ARMA(1,1)={arma_aic:.1f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why can ARMA models be more parsimonious than pure AR or MA models?</summary>

<div class="answer">
<strong>Answer:</strong> Many real processes have both autoregressive dynamics (momentum/persistence) and shock effects that dissipate over time. Modeling this with pure AR or MA requires many parameters, while ARMA captures both with fewer parameters.

<strong>Example:</strong> A process requiring AR(10) or MA(10) might be well-approximated by ARMA(1,1) with just 2 parameters.

**Key insight:** ARMA(1,1) has infinite ACF decay (like AR(∞)) and infinite PACF decay (like MA(∞)), achieving complex correlation structure parsimoniously.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming more parameters is better. ARMA(3,3) often overfits. Start simple—ARMA(1,1) is frequently sufficient.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What is parameter redundancy in ARMA models and why is it problematic?</summary>

<div class="answer">
<strong>Answer:</strong> Parameter redundancy (common factor problem) occurs when AR and MA polynomials share a root, causing them to cancel. The model reduces to a lower-order ARMA.

<strong>Example:</strong>
$$(1 - 0.5L)X_t = (1 - 0.5L)\epsilon_t$$

Both sides have factor $(1-0.5L)$. Canceling gives $X_t = \epsilon_t$.

**Problems:**
1. Extra parameters don't improve fit
2. Estimation becomes unstable (nearly singular Hessian)
3. Standard errors explode
4. Misleading model complexity

**Detection:** Check if AR and MA roots are close. Large standard errors suggest near-redundancy.

<div class="pitfall">
<strong>Common pitfall:</strong> Not checking for common factors. Software may fit ARMA(2,2) when ARMA(1,1) suffices, leading to unstable estimates.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> For ARMA(1,1), show that the ACF follows $\rho(h) = \phi^{h-1}\rho(1)$ for $h \geq 1$.</summary>

<div class="answer">
<strong>Answer:</strong> For $h \geq 2$, the autocovariance satisfies the AR(1) recursion: $\gamma(h) = \phi\gamma(h-1)$.

<strong>Derivation:</strong>
Multiply both sides of $X_t = \phi X_{t-1} + \epsilon_t + \theta\epsilon_{t-1}$ by $X_{t-h}$ and take expectations:

For $h \geq 2$:
$$E[X_t X_{t-h}] = \phi E[X_{t-1}X_{t-h}] + E[\epsilon_t X_{t-h}] + \theta E[\epsilon_{t-1}X_{t-h}]$$

Since $\epsilon_t$ and $\epsilon_{t-1}$ are uncorrelated with $X_{t-h}$ when $h \geq 2$:
$$\gamma(h) = \phi\gamma(h-1)$$

Therefore:
$$\gamma(h) = \phi^{h-1}\gamma(1)$$
$$\rho(h) = \phi^{h-1}\rho(1)$$

**Note:** $\rho(1)$ itself is not simply $\phi$ — it depends on both $\phi$ and $\theta$.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming ARMA(1,1) ACF at lag 1 equals $\phi$. The MA component modifies $\rho(1)$; only subsequent lags follow pure AR(1) decay.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Derive the condition for an ARMA(1,1) to be both stationary and invertible.</summary>

<div class="answer">
<strong>Answer:</strong> Stationarity requires $|\phi| < 1$; invertibility requires $|\theta| < 1$.

<strong>Derivation:</strong>

**Stationarity:**
AR polynomial: $\Phi(z) = 1 - \phi z$
Root: $z = 1/\phi$
Outside unit circle: $|1/\phi| > 1 \Rightarrow |\phi| < 1$

**Invertibility:**
MA polynomial: $\Theta(z) = 1 + \theta z$
Root: $z = -1/\theta$
Outside unit circle: $|{-1/\theta}| > 1 \Rightarrow |\theta| < 1$

**Combined:** The process is stationary and invertible iff $|\phi| < 1$ AND $|\theta| < 1$.

<div class="pitfall">
<strong>Common pitfall:</strong> Checking only stationarity. A stationary but non-invertible ARMA has improper AR(∞) representation, causing forecasting and diagnostic issues.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You fit several models and get: AR(2) AIC=520, MA(2) AIC=525, ARMA(1,1) AIC=515, ARMA(2,1) AIC=514. Which model do you choose and why?</summary>

<div class="answer">
<strong>Answer:</strong> Choose ARMA(1,1) despite ARMA(2,1) having slightly lower AIC.

<strong>Reasoning:</strong>
1. AIC difference of 1 is negligible (within noise)
2. Parsimony principle: simpler model preferred when performance similar
3. ARMA(1,1) is more stable and interpretable
4. Additional AR parameter unlikely to improve forecasts

**Decision framework:**
- AIC difference < 2: models essentially equivalent
- AIC difference 2-7: some evidence for lower AIC model
- AIC difference > 10: strong evidence for lower AIC model

Also consider:
- BIC (penalizes complexity more)
- Out-of-sample forecast accuracy
- Residual diagnostics for all candidates

<div class="pitfall">
<strong>Common pitfall:</strong> Blindly choosing lowest AIC. Small AIC differences are not meaningful. Always consider parsimony and validate with holdout data.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 4.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 4.
3. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley. Chapter 2.
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. Chapter 3.
