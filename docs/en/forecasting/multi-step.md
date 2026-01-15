# Multi-step Forecasting

<div class="interview-summary">
<strong>Interview Summary:</strong> Multi-step forecasting predicts multiple future values. Three strategies: recursive (iterate 1-step), direct (separate model per horizon), MIMO (multiple-input-multiple-output). Recursive accumulates errors but uses single model; direct avoids error accumulation but needs h models. For ARIMA, recursive is standard. For ML, direct often preferred.
</div>

## Core Definitions

**Multi-step Forecast**: Predict $y_{T+1}, y_{T+2}, \ldots, y_{T+H}$ given $y_1, \ldots, y_T$.

**Forecast Horizon (H)**: Number of steps ahead to predict.

**Strategies:**

1. **Recursive (Iterated)**: Use 1-step model repeatedly, feeding predictions as inputs
2. **Direct**: Train separate model for each horizon h
3. **MIMO**: Single model outputs all horizons simultaneously
4. **DirRec**: Hybrid of direct and recursive

## Math and Derivations

### Recursive Strategy

Train model: $\hat{y}_{t+1} = f(y_t, y_{t-1}, \ldots)$

For h-step forecast:
$$\hat{y}_{T+1} = f(y_T, y_{T-1}, \ldots)$$
$$\hat{y}_{T+2} = f(\hat{y}_{T+1}, y_T, \ldots)$$
$$\hat{y}_{T+h} = f(\hat{y}_{T+h-1}, \hat{y}_{T+h-2}, \ldots)$$

**Properties:**
- Uses single model
- Consistent with underlying DGP
- Error accumulates through iterations

### Direct Strategy

Train h separate models:
$$\hat{y}_{t+h}^{(h)} = f_h(y_t, y_{t-1}, \ldots)$$

Each model directly predicts h steps ahead.

**Properties:**
- No error propagation
- Requires h models
- Each model trained on different target
- May violate consistency across horizons

### Error Analysis

**Recursive error:**
$$e_{T+h}^{rec} = \sum_{j=1}^{h}\alpha_j\epsilon_{T+j} + O(\text{model error})$$

Error compounds through iterations.

**Direct error:**
$$e_{T+h}^{dir} = \epsilon_{T+h}^{(h)} + O(\text{model error}_h)$$

No compounding, but model $f_h$ may be less efficient.

### Theoretical Comparison

**Theorem (Ben Taieb & Hyndman):**
Under correct model specification:
- Recursive is optimal (MSFE-minimizing)
- Direct is consistent but less efficient

Under misspecification:
- Direct may outperform recursive
- Recursive compounds misspecification errors

## Algorithm/Model Sketch

**Strategy Selection Guidelines:**

```
IF model is well-specified (ARIMA, ETS):
   USE recursive
   - Theoretical optimality
   - Proper uncertainty quantification

ELIF using ML/nonparametric methods:
   USE direct
   - Avoids error accumulation
   - Each horizon optimized separately

ELIF forecast horizons are related:
   USE MIMO
   - Single model, multiple outputs
   - Can capture horizon dependencies

FOR robust approach:
   COMBINE recursive and direct
   - Average forecasts
   - Often improves accuracy
```

**MIMO Implementation:**
```python
# Train model to predict H horizons at once
# Input: features X
# Output: [y_{t+1}, y_{t+2}, ..., y_{t+H}]

X_train, Y_train = create_mimo_data(y, lags=p, horizon=H)
model = MultiOutputRegressor(base_model)
model.fit(X_train, Y_train)

# Forecast
X_new = get_features(y[-p:])
forecasts = model.predict(X_new)  # Returns [ŷ_{T+1}, ..., ŷ_{T+H}]
```

## Common Pitfalls

1. **Using recursive with ML**: Tree-based models don't extrapolate well; recursive strategy can produce flat or exploding forecasts.

2. **Ignoring error accumulation**: For long horizons, recursive ARIMA uncertainty grows. Don't trust tight intervals at h=100.

3. **Direct model inconsistency**: Direct models for h=5 and h=6 may give $\hat{y}_{T+6} < \hat{y}_{T+5}$ (non-monotonic when trend expected).

4. **Computational cost**: Direct requires H models. For H=365 (daily data, 1 year), this is expensive.

5. **Different targets, same features**: Direct models at different horizons have different optimal features. Using same features for all h is suboptimal.

6. **Ignoring seasonality in direct**: Direct model for h=12 on monthly data should capture annual pattern, but training data may not provide enough signal.

## Mini Example

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def create_lagged_data(y, lags, horizon):
    """Create dataset for direct/MIMO forecasting."""
    X, Y = [], []
    for t in range(lags, len(y) - horizon):
        X.append(y[t-lags:t][::-1])  # [y_{t-1}, y_{t-2}, ...]
        Y.append(y[t:t+horizon])      # [y_t, y_{t+1}, ...]
    return np.array(X), np.array(Y)

# Generate AR(2) data
np.random.seed(42)
n = 500
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.5 * y[t-1] + 0.3 * y[t-2] + np.random.randn()

# Split
train, test = y[:400], y[400:]
H = 10  # Forecast horizon

# Recursive strategy
from statsmodels.tsa.ar_model import AutoReg
model_rec = AutoReg(train, lags=2).fit()
forecast_rec = model_rec.forecast(H)

# Direct strategy
X_train, Y_train = create_lagged_data(train, lags=5, horizon=H)
direct_models = [Ridge().fit(X_train, Y_train[:, h]) for h in range(H)]
X_new = train[-5:][::-1].reshape(1, -1)
forecast_dir = np.array([m.predict(X_new)[0] for m in direct_models])

# MIMO strategy
mimo_model = MultiOutputRegressor(Ridge()).fit(X_train, Y_train)
forecast_mimo = mimo_model.predict(X_new)[0]

# Compare
print("Forecasts comparison:")
print(f"Recursive: {np.round(forecast_rec[:5], 2)}")
print(f"Direct:    {np.round(forecast_dir[:5], 2)}")
print(f"MIMO:      {np.round(forecast_mimo[:5], 2)}")
print(f"Actual:    {np.round(test[:5], 2)}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why does the recursive strategy accumulate errors while direct does not?</summary>

<div class="answer">
<strong>Answer:</strong>

**Recursive:** At each step, uses predicted values as inputs:
$$\hat{y}_{T+2} = f(\hat{y}_{T+1}, y_T, \ldots)$$

Error in $\hat{y}_{T+1}$ affects $\hat{y}_{T+2}$, which affects $\hat{y}_{T+3}$, etc.

**Direct:** Each horizon uses only actual observed values:
$$\hat{y}_{T+h} = f_h(y_T, y_{T-1}, \ldots)$$

Errors at different horizons are independent (given the data).

**Trade-off:**
- Recursive: consistent but error compounds
- Direct: no compounding but less efficient (separate models)

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking direct is always better. For well-specified models, recursive is theoretically optimal. Direct wins mainly under misspecification.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> When would you prefer direct over recursive forecasting?</summary>

<div class="answer">
<strong>Answer:</strong> Prefer direct when:

1. **Model misspecification**: If 1-step model is wrong, recursive compounds errors
2. **ML methods**: Trees/NNs often perform poorly in recursive mode
3. **Horizon-specific patterns**: Different dynamics at different horizons
4. **Long horizons**: Recursive uncertainty explodes; direct stays bounded
5. **Non-stationary features**: Recursive may drift; direct anchors to data

Prefer recursive when:
- Model is well-specified (ARIMA, ETS)
- Need consistent probability framework
- Computational efficiency matters
- Understanding model dynamics is important

<div class="pitfall">
<strong>Common pitfall:</strong> Using recursive with gradient boosting — trees don't extrapolate, leading to flat/constant long-horizon forecasts.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> For AR(1) with $y_t = \phi y_{t-1} + \epsilon_t$, derive the h-step recursive forecast and show it equals the direct optimal forecast.</summary>

<div class="answer">
<strong>Answer:</strong>

**Recursive:**
$$\hat{y}_{T+1|T} = \phi y_T$$
$$\hat{y}_{T+2|T} = \phi \hat{y}_{T+1|T} = \phi^2 y_T$$
$$\hat{y}_{T+h|T} = \phi^h y_T$$

**Direct (optimal):**
The conditional expectation:
$$E[y_{T+h}|y_T] = E[\phi^h y_T + \sum_{j=0}^{h-1}\phi^j\epsilon_{T+h-j}|y_T]$$
$$= \phi^h y_T + 0 = \phi^h y_T$$

They're identical! For correctly specified linear models, recursive = direct optimal.

**Key insight:** When model is correct, feeding $\hat{y}_{T+j}$ in place of $y_{T+j}$ gives the same result as computing $E[y_{T+h}|y_{1:T}]$ directly.

<div class="pitfall">
<strong>Common pitfall:</strong> This equivalence holds for linear models. For nonlinear models, recursive ≠ direct even when correctly specified.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How does the forecast error variance differ between recursive and direct strategies?</summary>

<div class="answer">
<strong>Answer:</strong>

**Recursive (correct model):**
$$\text{Var}(e_{T+h}^{rec}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2$$

This is the theoretical minimum (Cramér-Rao bound for linear prediction).

**Direct (correct model):**
$$\text{Var}(e_{T+h}^{dir}) = \text{Var}(e_{T+h}^{rec}) + \text{estimation variance}_h$$

Direct adds variance because model $f_h$ is estimated less efficiently than the 1-step model (less data effectively used).

**Under misspecification:**
- Recursive: $\text{Var}(e_{T+h}^{rec}) \approx h \times \text{bias}^2 + \text{variance}$
- Direct: $\text{Var}(e_{T+h}^{dir}) \approx \text{bias}_h^2 + \text{variance}_h$

Direct doesn't compound bias across horizons.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming direct always has larger variance. Under misspecification, direct often wins because it avoids compounding the bias.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You need to forecast daily sales 30 days ahead. You have an XGBoost model. Which strategy do you use?</summary>

<div class="answer">
<strong>Answer:</strong> **Direct** or **MIMO** strategy.

**Why not recursive:**
- XGBoost is a tree-based model that doesn't extrapolate
- Recursive feeding predictions back leads to:
  - Forecasts that flatten to mean
  - Or erratic behavior if predictions drift outside training range
- 30-step recursion compounds errors significantly

**Recommended approach:**
1. **Direct:** Train 30 separate XGBoost models
   - Horizon-specific optimization
   - Can use different features per horizon
   - Computationally more expensive

2. **MIMO:** Train one multi-output model
   - Use `MultiOutputRegressor(XGBRegressor())`
   - Or custom multi-output architecture
   - More efficient than 30 models

3. **Hybrid:** Use LightGBM/XGBoost for short horizons, average with simpler model for longer horizons

<div class="pitfall">
<strong>Common pitfall:</strong> Using recursive XGBoost — forecasts often degrade to constant or oscillate. Always validate multi-step behavior before production.
</div>
</div>
</details>

## References

1. Ben Taieb, S., & Hyndman, R. J. (2014). A gradient boosting approach to the Kaggle load forecasting competition. *IJF*, 30(2), 382-394.
2. Chevillon, G. (2007). Direct multi-step estimation and forecasting. *Journal of Economic Surveys*, 21(4), 746-785.
3. Marcellino, M., Stock, J. H., & Watson, M. W. (2006). A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series. *Journal of Econometrics*, 135(1-2), 499-526.
4. Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition. *Expert Systems with Applications*, 39(8), 7067-7083.
