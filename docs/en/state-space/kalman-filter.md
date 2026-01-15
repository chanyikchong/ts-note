# State Space Models and Kalman Filter

<div class="interview-summary">
<strong>Interview Summary:</strong> State space models represent time series via hidden states evolving over time. Kalman filter recursively estimates states: prediction step (propagate state forward), update step (incorporate new observation). Optimal for linear Gaussian systems. Core equations: $x_t = Fx_{t-1} + w_t$, $y_t = Hx_t + v_t$. Unifies ARIMA, exponential smoothing, and structural models.
</div>

## Core Definitions

**State Space Representation:**

State equation: $\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t$, $\mathbf{w}_t \sim N(0, \mathbf{Q})$

Observation equation: $y_t = \mathbf{H}\mathbf{x}_t + v_t$, $v_t \sim N(0, R)$

**Components:**
- $\mathbf{x}_t$: State vector (unobserved)
- $y_t$: Observation (data)
- $\mathbf{F}$: State transition matrix
- $\mathbf{H}$: Observation matrix
- $\mathbf{Q}$: State noise covariance
- $R$: Observation noise variance

**Local Level Model (simplest):**
$$\mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)$$
$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2_\epsilon)$$

## Math and Derivations

### Kalman Filter Recursions

**Prediction Step:**
$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}' + \mathbf{Q}$$

**Update Step:**
$$\mathbf{K}_t = \mathbf{P}_{t|t-1}\mathbf{H}'(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}' + R)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(y_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t\mathbf{H})\mathbf{P}_{t|t-1}$$

**Key quantities:**
- $\hat{\mathbf{x}}_{t|t}$: Filtered state estimate (using data up to t)
- $\mathbf{P}_{t|t}$: State covariance (uncertainty)
- $\mathbf{K}_t$: Kalman gain (weight on new observation)
- $v_t = y_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1}$: Innovation (prediction error)

### Local Level Model Kalman Filter

State: $\mathbf{x}_t = \mu_t$ (scalar)
Matrices: $F = 1$, $H = 1$, $Q = \sigma^2_\eta$, $R = \sigma^2_\epsilon$

Recursions:
$$\hat{\mu}_{t|t-1} = \hat{\mu}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + \sigma^2_\eta$$
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma^2_\epsilon}$$
$$\hat{\mu}_{t|t} = \hat{\mu}_{t|t-1} + K_t(y_t - \hat{\mu}_{t|t-1})$$
$$P_{t|t} = (1 - K_t)P_{t|t-1}$$

As $t \to \infty$: $K_t \to K^* = $ steady-state gain (related to SES α).

### Local Linear Trend Model

State: $\mathbf{x}_t = (\mu_t, \beta_t)'$ (level and trend)

$$\begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} \mu_{t-1} \\ \beta_{t-1} \end{pmatrix} + \begin{pmatrix} \eta_t \\ \zeta_t \end{pmatrix}$$

$$y_t = (1, 0)\begin{pmatrix} \mu_t \\ \beta_t \end{pmatrix} + \epsilon_t$$

## Algorithm/Model Sketch

**Kalman Filter Algorithm:**

```
Input: y[1:n], F, H, Q, R, x0, P0
Output: filtered states, innovations, likelihood

Initialize:
  x_hat = x0
  P = P0
  log_lik = 0

For t = 1 to n:
  # Prediction
  x_pred = F @ x_hat
  P_pred = F @ P @ F' + Q

  # Innovation
  v = y[t] - H @ x_pred
  S = H @ P_pred @ H' + R

  # Update
  K = P_pred @ H' @ inv(S)
  x_hat = x_pred + K @ v
  P = (I - K @ H) @ P_pred

  # Likelihood contribution
  log_lik += -0.5 * (log(det(S)) + v' @ inv(S) @ v)

Return x_hat_history, P_history, log_lik
```

**Kalman Smoother** (uses all data):
$$\hat{\mathbf{x}}_{t|n} = \hat{\mathbf{x}}_{t|t} + \mathbf{J}_t(\hat{\mathbf{x}}_{t+1|n} - \hat{\mathbf{x}}_{t+1|t})$$

where $\mathbf{J}_t = \mathbf{P}_{t|t}\mathbf{F}'\mathbf{P}_{t+1|t}^{-1}$

## Common Pitfalls

1. **Numerical instability**: Covariance matrices can become non-positive-definite. Use square-root or UD factorization.

2. **Wrong initialization**: Poor $\mathbf{x}_0, \mathbf{P}_0$ affects early estimates. Use diffuse initialization for unknown initial states.

3. **Model misspecification**: Kalman filter is optimal only for true model. Non-Gaussian or nonlinear systems need extensions (EKF, UKF, particle filter).

4. **Forgetting the smoother**: For historical analysis (not real-time), use Kalman smoother to incorporate future observations.

5. **Over-complicated state space**: Can represent many models but simple alternatives (ARIMA) may be easier.

6. **Confusing filter vs. forecast**: Filter = estimate given data up to t. Forecast = prediction beyond observed data.

## Mini Example

```python
import numpy as np

def kalman_filter_local_level(y, sigma_eta, sigma_eps, mu0=None, P0=1000):
    """Kalman filter for local level model."""
    n = len(y)

    # Initialize
    mu_filt = np.zeros(n)
    P_filt = np.zeros(n)
    mu_pred = mu0 if mu0 is not None else y[0]
    P_pred = P0

    for t in range(n):
        # Prediction (for t > 0)
        if t > 0:
            mu_pred = mu_filt[t-1]
            P_pred = P_filt[t-1] + sigma_eta**2

        # Update
        K = P_pred / (P_pred + sigma_eps**2)
        mu_filt[t] = mu_pred + K * (y[t] - mu_pred)
        P_filt[t] = (1 - K) * P_pred

    return mu_filt, P_filt

# Generate local level data
np.random.seed(42)
n = 100
sigma_eta, sigma_eps = 0.5, 1.0

# True states
mu_true = np.cumsum(np.random.randn(n) * sigma_eta)
# Observations
y = mu_true + np.random.randn(n) * sigma_eps

# Run Kalman filter
mu_hat, P_hat = kalman_filter_local_level(y, sigma_eta, sigma_eps)

print("Kalman Filter Results:")
print(f"Final state estimate: {mu_hat[-1]:.2f}")
print(f"True final state: {mu_true[-1]:.2f}")
print(f"Final state std: {np.sqrt(P_hat[-1]):.3f}")

# Steady-state Kalman gain
K_steady = (-sigma_eps**2 + np.sqrt(sigma_eps**4 + 4*sigma_eta**2*sigma_eps**2)) / (2*sigma_eps**2)
print(f"\nSteady-state Kalman gain: {K_steady:.3f}")
print(f"Equivalent SES alpha: {K_steady:.3f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the intuition behind the Kalman gain?</summary>

<div class="answer">
<strong>Answer:</strong> The Kalman gain $K_t$ balances trust in prediction vs. trust in observation.

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R} = \frac{\text{prediction uncertainty}}{\text{prediction uncertainty} + \text{observation noise}}$$

**Interpretation:**
- High $K_t$ (close to 1): Observation is trusted more; state updates significantly
- Low $K_t$ (close to 0): Prediction is trusted more; observation has little effect

**When is K high?**
- High state uncertainty ($P$ large)
- Low observation noise ($R$ small)

**When is K low?**
- Low state uncertainty ($P$ small)
- High observation noise ($R$ large)

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking gain is fixed. It evolves as uncertainty changes. Early on, gain may be high (uncertain prior); later, it stabilizes as filter "learns."
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> How does the local level model relate to simple exponential smoothing?</summary>

<div class="answer">
<strong>Answer:</strong> They're equivalent! Local level model with specific noise ratio produces SES.

**Local level:**
$$\mu_t = \mu_{t-1} + \eta_t, \quad y_t = \mu_t + \epsilon_t$$

**Kalman update:**
$$\hat{\mu}_{t|t} = \hat{\mu}_{t|t-1} + K(y_t - \hat{\mu}_{t|t-1})$$

**At steady state:** $K \to K^* = \alpha$ (SES smoothing parameter)

The relationship:
$$\alpha = \frac{-\sigma_\epsilon^2 + \sqrt{\sigma_\epsilon^4 + 4\sigma_\eta^2\sigma_\epsilon^2}}{2\sigma_\eta^2}$$

**Key insight:** SES is the steady-state Kalman filter for local level model. Kalman provides:
- Optimal initialization
- Proper uncertainty quantification
- Connection to likelihood

<div class="pitfall">
<strong>Common pitfall:</strong> Using SES without recognizing it assumes specific state space structure. If dynamics differ, SES may be suboptimal.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the steady-state Kalman gain for the local level model.</summary>

<div class="answer">
<strong>Answer:</strong> At steady state, $P_{t|t} = P^*$ (constant).

**Recursion:**
$$P_{t|t-1} = P_{t-1|t-1} + Q = P^* + Q$$
$$P_{t|t} = (1 - K_t)P_{t|t-1} = P^*$$

**Steady state condition:**
$$P^* = (1 - K^*)(P^* + Q)$$
$$P^* = P^* + Q - K^*(P^* + Q)$$
$$K^* = \frac{Q}{P^* + Q}$$

Also: $K^* = \frac{P^* + Q}{P^* + Q + R}$

Solving for $P^*$:
$$P^* = \frac{-R + \sqrt{R^2 + 4QR}}{2}$$

And:
$$K^* = \frac{-R + \sqrt{R^2 + 4QR}}{2Q}$$

For Q = $\sigma_\eta^2$, R = $\sigma_\epsilon^2$, this gives the SES α relationship.

<div class="pitfall">
<strong>Common pitfall:</strong> Steady state may take many iterations to reach. For short series, time-varying gain matters.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> What is the difference between filtered and smoothed state estimates?</summary>

<div class="answer">
<strong>Answer:</strong>

**Filtered:** $\hat{x}_{t|t} = E[x_t | y_1, \ldots, y_t]$
- Uses observations up to time t
- Real-time estimate
- From forward Kalman pass

**Smoothed:** $\hat{x}_{t|n} = E[x_t | y_1, \ldots, y_n]$
- Uses all observations
- Retrospective estimate
- Requires backward pass after forward

**Key difference:** Smoother uses future observations to refine past state estimates.

**When to use which:**
- Real-time forecasting: Filter
- Historical analysis: Smoother
- Parameter estimation: Smoother (better likelihood)

**Variance relationship:**
$$\text{Var}(\hat{x}_{t|n}) \leq \text{Var}(\hat{x}_{t|t})$$

Smoother is never worse than filter.

<div class="pitfall">
<strong>Common pitfall:</strong> Using filter estimates for historical analysis when smoother is available and more accurate.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You have GPS measurements with known accuracy (R) but unknown vehicle dynamics (Q). How would you tune the Kalman filter?</summary>

<div class="answer">
<strong>Answer:</strong>

**Approach 1: Maximum Likelihood**
- Treat Q as parameter
- Compute likelihood from innovations: $\log L = -\frac{1}{2}\sum(\log S_t + v_t^2/S_t)$
- Optimize Q to maximize likelihood

**Approach 2: Innovation-based tuning**
- Innovations $v_t$ should be white noise with variance $S_t$
- If innovations autocorrelated: Q too small
- If innovation variance >> $S_t$: Q too small
- If innovation variance << $S_t$: Q too large

**Approach 3: Adaptive filtering**
- Estimate Q online from recent innovation statistics
- $\hat{Q}_t = $ sample variance of recent innovations minus R

**Practical steps:**
1. Start with Q = R (equal trust)
2. Check innovation statistics
3. Grid search or gradient optimization
4. Validate on holdout data

<div class="pitfall">
<strong>Common pitfall:</strong> Setting Q too small makes filter overconfident and slow to track changes. Too large makes it noisy. Cross-validate!
</div>
</div>
</details>

## References

1. Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
2. Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
3. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
4. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 13.
