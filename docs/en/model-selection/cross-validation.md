# Cross-Validation for Time Series

<div class="interview-summary">
<strong>Interview Summary:</strong> Standard k-fold CV breaks temporal order and causes data leakage. Time series CV uses rolling/expanding windows: train on past, test on future. Key methods: rolling origin, blocked CV, h-step-ahead CV. Always respect temporal order. CV estimates out-of-sample error; use for model selection and hyperparameter tuning.
</div>

## Core Definitions

**Time Series Cross-Validation:**
Evaluate model performance by repeatedly:
1. Training on past data
2. Testing on future data (never seen during training)
3. Rolling the window forward

**Rolling Origin Evaluation:**
```
Train: [1, ..., t]     → Test: [t+1, ..., t+h]
Train: [1, ..., t+1]   → Test: [t+2, ..., t+h+1]
...
Train: [1, ..., T-h]   → Test: [T-h+1, ..., T]
```

**Expanding Window:** Training set grows; uses all past data.

**Sliding Window:** Training set is fixed size; drops oldest data.

## Math and Derivations

### Rolling Origin Forecast Error

For origin $t$ and horizon $h$:
$$e_{t+h|t} = y_{t+h} - \hat{y}_{t+h|t}$$

Average over all origins:
$$\text{RMSE}(h) = \sqrt{\frac{1}{T-t_0-h+1}\sum_{t=t_0}^{T-h}e_{t+h|t}^2}$$

### Forecast Accuracy Metrics

**MAE (Mean Absolute Error):**
$$\text{MAE} = \frac{1}{n}\sum|e_t|$$

**RMSE (Root Mean Squared Error):**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum e_t^2}$$

**MAPE (Mean Absolute Percentage Error):**
$$\text{MAPE} = \frac{100}{n}\sum\left|\frac{e_t}{y_t}\right|$$

**SMAPE (Symmetric MAPE):**
$$\text{sMAPE} = \frac{200}{n}\sum\frac{|e_t|}{|y_t| + |\hat{y}_t|}$$

**MASE (Mean Absolute Scaled Error):**
$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{n-1}\sum_{t=2}^{n}|y_t - y_{t-1}|}$$

MASE < 1 means better than naive forecast.

### Why Standard CV Fails

Standard k-fold CV:
- Randomly splits data
- Training fold may contain future observations
- Test fold may contain past observations

This causes **data leakage**: model sees future information during training, giving optimistic error estimates.

## Algorithm/Model Sketch

**Rolling Origin CV:**

```python
def rolling_origin_cv(y, model_fn, min_train, horizon, step=1):
    """
    y: time series
    model_fn: function that fits model and returns forecasts
    min_train: minimum training size
    horizon: forecast horizon
    step: how much to move origin each iteration
    """
    errors = []

    for t in range(min_train, len(y) - horizon, step):
        # Train on [0:t], test on [t:t+horizon]
        train = y[:t]
        test = y[t:t+horizon]

        # Fit and forecast
        forecast = model_fn(train, horizon)

        # Store errors
        errors.append(test - forecast)

    return np.array(errors)
```

**Blocked CV (for related series):**
```
Fold 1: Train [blocks 2,3,4,5] → Test [block 1]
Fold 2: Train [blocks 1,3,4,5] → Test [block 2]
...
```

Blocks are contiguous time periods. Less ideal but useful when multiple series share parameters.

## Common Pitfalls

1. **Using standard k-fold CV**: Breaks temporal order, causes leakage. Never use for time series.

2. **Testing on training period**: Even with rolling origin, some implementations accidentally include overlapping data.

3. **Ignoring horizon**: CV for h=1 doesn't guarantee good h=10 performance. Match CV horizon to application.

4. **Fixed origin only**: Testing from single origin underestimates variance. Use multiple origins.

5. **Computation cost**: Full rolling CV with refitting is expensive. Consider step > 1 or fixed models.

6. **Non-representative windows**: If dynamics change, old data may mislead. Consider sliding window.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def mase(actual, forecast, train):
    """Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(np.diff(train)))
    return mae / naive_mae

# Generate data
np.random.seed(42)
n = 200
y = np.cumsum(np.random.randn(n)) + 0.1 * np.arange(n)

# Rolling origin CV
min_train = 100
horizon = 5
step = 5

results = {'ARIMA(1,1,0)': [], 'ARIMA(1,1,1)': [], 'ARIMA(2,1,1)': []}

for t in range(min_train, len(y) - horizon, step):
    train = y[:t]
    test = y[t:t+horizon]

    for name, order in [('ARIMA(1,1,0)', (1,1,0)),
                        ('ARIMA(1,1,1)', (1,1,1)),
                        ('ARIMA(2,1,1)', (2,1,1))]:
        try:
            model = ARIMA(train, order=order).fit()
            forecast = model.forecast(horizon)
            error = mase(test, forecast, train)
            results[name].append(error)
        except:
            results[name].append(np.nan)

# Compare models
print("Cross-Validation Results (MASE):")
for name, errors in results.items():
    valid_errors = [e for e in errors if not np.isnan(e)]
    print(f"  {name}: Mean={np.mean(valid_errors):.3f}, "
          f"Std={np.std(valid_errors):.3f}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why does standard k-fold cross-validation fail for time series?</summary>

<div class="answer">
<strong>Answer:</strong> Standard k-fold CV randomly assigns observations to folds, breaking temporal order. This causes:

1. **Data leakage**: Training data may include future observations relative to test data
2. **Unrealistic evaluation**: In practice, you never have future data to train on
3. **Optimistic error estimates**: Model implicitly learns from future, inflating apparent accuracy
4. **Autocorrelation ignored**: Nearby points in train and test are correlated, reducing effective test independence

**Example:** If test fold contains y[50:60] and train contains y[55:100], the model uses y[55:60] (future!) during training.

<div class="pitfall">
<strong>Common pitfall:</strong> Using sklearn's `cross_val_score` directly on time series. Always use `TimeSeriesSplit` or custom rolling evaluation.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What's the difference between expanding window and sliding window CV?</summary>

<div class="answer">
<strong>Answer:</strong>

**Expanding window:**
- Training set grows: [1:t], [1:t+1], [1:t+2], ...
- Uses all historical data
- Better for stable processes
- More data → lower variance estimates

**Sliding window:**
- Training set is fixed size: [t-w:t], [t-w+1:t+1], ...
- Drops oldest data
- Better for non-stationary/evolving processes
- Adapts to recent patterns

**Choice depends on:**
- Stationarity: non-stationary → sliding
- Data availability: limited → expanding
- Concept drift: present → sliding
- Computational cost: sliding is more expensive (always refits)

<div class="pitfall">
<strong>Common pitfall:</strong> Using expanding window when dynamics change. Old data misleads the model. Check for structural breaks.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Why is MASE preferred over MAPE for forecast evaluation?</summary>

<div class="answer">
<strong>Answer:</strong> MASE advantages:

1. **Scale-independent**: Like MAPE, but handles zeros
2. **No division by zero**: MAPE fails when $y_t = 0$
3. **Symmetric**: Doesn't favor under/over-prediction asymmetrically (unlike MAPE)
4. **Benchmark comparison**: MASE < 1 means better than naive
5. **Well-defined for intermittent series**: Common in demand forecasting

**Formula:**
$$\text{MASE} = \frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$$

where MAE_naive uses seasonal naive or 1-step naive as benchmark.

**MAPE problems:**
- Infinite when $y_t = 0$
- Asymmetric: 50% error on y=100 (predict 50 or 150) treated differently
- Scale-dependent interpretation

<div class="pitfall">
<strong>Common pitfall:</strong> Using MAPE for intermittent demand or data with zeros — gives undefined or misleading results.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How do you choose the minimum training size for rolling origin CV?</summary>

<div class="answer">
<strong>Answer:</strong> Balance between:

1. **Statistical requirements:**
   - Need enough data for reliable estimation
   - Rule: at least 3-5 observations per parameter
   - For ARIMA(p,d,q): min ~50 + 10(p+q) observations

2. **Practical considerations:**
   - More training → better model estimates
   - But also → fewer CV folds → higher variance of CV estimate
   - Typical: 60-80% of data for first training set

3. **Domain knowledge:**
   - If dynamics change, recent data matters more
   - Full business cycles should be included (e.g., full year for seasonal)

**Formula guidance:**
$$\text{min\_train} = \max(50, 2 \times m, 5k + 10)$$

where m = seasonal period, k = number of parameters.

<div class="pitfall">
<strong>Common pitfall:</strong> Using too small min_train gives unreliable early models; using too large leaves few CV folds for variance estimation.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You have 3 years of daily data (1095 observations) and need to select a model for 7-day forecasting. Design a CV scheme.</summary>

<div class="answer">
<strong>Answer:</strong>

**Recommended scheme:**

```
min_train = 365 (1 full year to capture seasonality)
horizon = 7
step = 7 (weekly, reduces computation)
```

This gives: (1095 - 365 - 7) / 7 ≈ 103 CV folds

**Implementation:**
```python
for t in range(365, 1095-7, 7):
    train = data[:t]
    test = data[t:t+7]
    # Fit and evaluate
```

**Considerations:**
1. **Include full seasonality**: 365 days captures annual pattern
2. **Match horizon**: CV horizon = production horizon (7 days)
3. **Step = horizon**: Non-overlapping test sets for independence
4. **Metrics**: Use MASE, MAE, RMSE at each horizon h=1,...,7

**Variants:**
- Sliding window: train on last 365 days only (if non-stationary)
- Gap: skip 1-2 days between train/test to simulate production delay

<div class="pitfall">
<strong>Common pitfall:</strong> Using step=1 with 1095 observations → 723 fits, very slow. Use larger step for efficiency.
</div>
</div>
</details>

## References

1. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 5.
3. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF*, 16(4), 437-450.
4. Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109(11), 1997-2028.
