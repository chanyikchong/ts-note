# Moving Average (MA) Models

<div class="interview-summary">
<strong>Interview Summary:</strong> MA(q) models express current value as a linear combination of current and past q noise terms. MA processes are always stationary (finite linear combination of white noise). ACF cuts off after lag q; PACF decays exponentially. Estimation requires nonlinear optimization (MLE). Invertibility requires roots outside unit circle.
</div>

## Core Definitions

**MA(q) Model**:
$$X_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q}$$

where $\epsilon_t \sim WN(0, \sigma^2)$.

**Lag Operator Form**:
$$X_t = \mu + \Theta(L)\epsilon_t$$

where $\Theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q$.

**Characteristic Polynomial**:
$$\Theta(z) = 1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q$$

**Invertibility Condition**: All roots of $\Theta(z) = 0$ must lie outside the unit circle.

## Math and Derivations

### MA(1) Model: $X_t = \mu + \epsilon_t + \theta\epsilon_{t-1}$

**Mean**: $E[X_t] = \mu$

**Variance**:
$$\gamma(0) = \text{Var}(X_t) = \sigma^2(1 + \theta^2)$$

**Autocovariance at lag 1**:
$$\gamma(1) = E[(\epsilon_t + \theta\epsilon_{t-1})(\epsilon_{t+1} + \theta\epsilon_t)] = \theta\sigma^2$$

**Autocovariance at lag $h \geq 2$**: $\gamma(h) = 0$

**ACF**:
$$\rho(1) = \frac{\theta}{1+\theta^2}, \quad \rho(h) = 0 \text{ for } h \geq 2$$

**Note**: Maximum $|\rho(1)| = 0.5$ at $\theta = \pm 1$.

### MA(q) General ACF

$$\gamma(h) = \begin{cases} \sigma^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h} & 0 \leq h \leq q \\ 0 & h > q \end{cases}$$

where $\theta_0 = 1$.

$$\rho(h) = \frac{\sum_{j=0}^{q-h} \theta_j \theta_{j+h}}{\sum_{j=0}^{q} \theta_j^2}$$

### Invertibility and the AR(∞) Representation

An invertible MA(1) can be written as AR(∞):
$$X_t = \mu + \epsilon_t + \theta\epsilon_{t-1}$$

If $|\theta| < 1$:
$$\epsilon_t = \sum_{j=0}^{\infty}(-\theta)^j(X_{t-j} - \mu)$$

This gives:
$$X_t = \mu(1+\theta) - \theta X_{t-1} + \theta^2 X_{t-2} - \theta^3 X_{t-3} + \cdots + \epsilon_t$$

**Why invertibility matters**: Allows expressing shocks in terms of observables. Required for proper forecasting and model interpretation.

### PACF of MA(1)

The PACF of MA(1) decays exponentially:
$$\phi_{hh} = \frac{-(-\theta)^h(1-\theta^2)}{1-\theta^{2(h+1)}}$$

For large $h$: $\phi_{hh} \approx -(-\theta)^h$

## Algorithm/Model Sketch

**Estimation Methods:**

1. **Innovation Algorithm**: Recursive method to compute MA coefficients from autocovariances.

2. **Conditional Sum of Squares (CSS)**:
   - Set pre-sample $\epsilon$ values to zero
   - Minimize $\sum \epsilon_t^2$
   - Fast but may be biased

3. **Exact Maximum Likelihood (MLE)**:
   - Accounts for initial conditions
   - Uses Kalman filter or direct likelihood
   - Most efficient asymptotically

**Estimation Challenges:**
- MA estimation is nonlinear (unlike AR)
- Multiple local optima possible
- Need good starting values
- Invertibility constraints must be enforced

**Order Selection:**
```
1. Examine ACF - cutoff suggests MA order
2. If ACF cuts off after lag q, start with MA(q)
3. Fit candidate models
4. Compare AIC/BIC
5. Check residual ACF/PACF
```

## Common Pitfalls

1. **Parameter identification**: MA(1) with $\theta$ and MA(1) with $1/\theta$ give the same ACF! Always enforce invertibility to get unique solution.

2. **Estimation difficulty**: MA models are harder to estimate than AR. Poor starting values lead to convergence issues. Use method="innovations" or CSS for initial estimates.

3. **Confusing MA order with differencing**: Large negative spike at lag 1 in ACF after differencing often indicates over-differencing, not MA(1).

4. **Misinterpreting ACF cutoff**: "Cutoff" means abrupt drop to zero, not just decay. AR processes also show ACF patterns—check PACF to distinguish.

5. **Non-invertible estimates**: If estimated $|\theta| > 1$, the model is non-invertible. Either flip to $1/\theta$ or reconsider model specification.

6. **Ignoring the unit root boundary**: $\theta = -1$ or $\theta = 1$ are non-invertible. Near these values, standard inference breaks down.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

# Generate MA(2) process
np.random.seed(42)
n = 300
theta1, theta2 = 0.6, 0.3
eps = np.random.randn(n + 2)
X = np.zeros(n)

for t in range(n):
    X[t] = eps[t+2] + theta1*eps[t+1] + theta2*eps[t]

# Check ACF (should cut off after lag 2)
acf_values = acf(X, nlags=10)
print("ACF:", np.round(acf_values, 3))
# Expected: significant at lags 1, 2; near zero after

# Fit MA(2) model
model = ARIMA(X, order=(0, 0, 2)).fit()
print(f"True: theta1={theta1}, theta2={theta2}")
print(f"Estimated: theta1={model.maparams[0]:.3f}, theta2={model.maparams[1]:.3f}")

# Theoretical ACF for comparison
gamma0 = 1 + theta1**2 + theta2**2
rho1 = (theta1 + theta1*theta2) / gamma0
rho2 = theta2 / gamma0
print(f"Theoretical rho(1)={rho1:.3f}, rho(2)={rho2:.3f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why are MA processes always stationary regardless of parameter values?</summary>

<div class="answer">
<strong>Answer:</strong> MA(q) is a finite linear combination of white noise: $X_t = \mu + \sum_{j=0}^{q}\theta_j\epsilon_{t-j}$. The mean is constant ($\mu$), variance is $\sigma^2\sum\theta_j^2$ (constant), and autocovariance depends only on lag (not time).

<strong>Explanation:</strong>
Stationarity requires:
1. Constant mean: $E[X_t] = \mu$ ✓
2. Constant variance: $\text{Var}(X_t) = \sigma^2(1+\theta_1^2+\cdots+\theta_q^2)$ ✓
3. Autocovariance depends only on lag: $\gamma(h)$ doesn't depend on $t$ ✓

All conditions are satisfied for any finite $\theta$ values because white noise is stationary and finite linear combinations preserve stationarity.

<div class="pitfall">
<strong>Common pitfall:</strong> Confusing stationarity with invertibility. MA is always stationary but not always invertible. Invertibility is about the AR(∞) representation, not stationarity.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Explain the concept of invertibility. Why do we care about it?</summary>

<div class="answer">
<strong>Answer:</strong> Invertibility means we can express the unobservable shocks $\epsilon_t$ as a convergent function of past observables $X_t, X_{t-1}, \ldots$. We care because:
1. It ensures unique model identification
2. It enables computing residuals for diagnostics
3. It's needed for proper forecasting updates

<strong>Technical detail:</strong> For MA(1): $X_t = \epsilon_t + \theta\epsilon_{t-1}$

If $|\theta| < 1$: $\epsilon_t = X_t - \theta X_{t-1} + \theta^2 X_{t-2} - \cdots$ (converges)
If $|\theta| > 1$: the expansion diverges

**Key equation:** MA(q) is invertible iff all roots of $\Theta(z) = 0$ lie outside the unit circle.

<div class="pitfall">
<strong>Common pitfall:</strong> Models with $\theta$ and $1/\theta$ produce identical ACFs but different forecasts. Without enforcing invertibility, you might get the "wrong" model that performs poorly.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Show that for MA(1), the ACF satisfies $|\rho(1)| \leq 0.5$.</summary>

<div class="answer">
<strong>Answer:</strong> We have $\rho(1) = \frac{\theta}{1+\theta^2}$. Taking the derivative and setting to zero finds the maximum.

<strong>Derivation:</strong>
$$\frac{d\rho(1)}{d\theta} = \frac{(1+\theta^2) - \theta(2\theta)}{(1+\theta^2)^2} = \frac{1-\theta^2}{(1+\theta^2)^2}$$

Setting equal to zero: $\theta = \pm 1$

At $\theta = 1$: $\rho(1) = \frac{1}{1+1} = 0.5$
At $\theta = -1$: $\rho(1) = \frac{-1}{1+1} = -0.5$

As $\theta \to 0$: $\rho(1) \to 0$
As $|\theta| \to \infty$: $\rho(1) \to 0$

Therefore $|\rho(1)| \leq 0.5$ with equality at $\theta = \pm 1$.

<div class="pitfall">
<strong>Common pitfall:</strong> If you observe sample $|\hat{\rho}(1)| > 0.5$, it's unlikely to be pure MA(1). Consider AR(1) (which can have any $|\rho(1)| < 1$) or mixed ARMA.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Derive the variance $\gamma(0)$ for MA(2): $X_t = \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2}$.</summary>

<div class="answer">
<strong>Answer:</strong> $\gamma(0) = \sigma^2(1 + \theta_1^2 + \theta_2^2)$

<strong>Derivation:</strong>
$$\gamma(0) = \text{Var}(X_t) = \text{Var}(\epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2})$$

Since $\epsilon_t$, $\epsilon_{t-1}$, $\epsilon_{t-2}$ are independent:
$$= \text{Var}(\epsilon_t) + \theta_1^2\text{Var}(\epsilon_{t-1}) + \theta_2^2\text{Var}(\epsilon_{t-2})$$
$$= \sigma^2 + \theta_1^2\sigma^2 + \theta_2^2\sigma^2$$
$$= \sigma^2(1 + \theta_1^2 + \theta_2^2)$$

**General formula for MA(q):**
$$\gamma(0) = \sigma^2\sum_{j=0}^{q}\theta_j^2 \text{ where } \theta_0 = 1$$

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting that $\theta_0 = 1$ by convention. The "1" in the formula comes from the $\epsilon_t$ term (coefficient 1).
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You fit an MA(1) model and get $\hat{\theta} = 1.2$. What should you do?</summary>

<div class="answer">
<strong>Answer:</strong> The estimate $\hat{\theta} = 1.2$ is outside the invertibility region. Options:
1. Flip to the invertible form: use $\theta' = 1/1.2 = 0.833$
2. Re-fit with invertibility constraints enforced
3. Reconsider model specification (maybe ARMA is better)

<strong>Explanation:</strong>
MA(1) with $\theta = 1.2$ and MA(1) with $\theta = 0.833$ produce identical ACFs:
- $\rho(1) = \frac{1.2}{1+1.44} = \frac{0.833}{1+0.694} = 0.492$

But only $\theta = 0.833$ is invertible. When computing forecasts or residuals, the invertible form is needed.

**Action plan:**
1. Check if software enforces invertibility automatically
2. If not, manually transform: $\theta_{new} = 1/\hat{\theta}$
3. Adjust variance estimate: $\sigma^2_{new} = \hat{\sigma}^2 \cdot \hat{\theta}^2$

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring non-invertibility warnings. The non-invertible model will give poor forecasts because the AR(∞) expansion diverges.
</div>
</div>
</details>

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley. Chapter 4.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 4.
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 3.
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. Chapter 3.
