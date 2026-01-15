# Practical Time Series Modeling

<div class="interview-summary">
<strong>Interview Summary:</strong> Practical modeling involves: proper train/test splits (temporal, never random), backtesting with rolling windows, handling deployment (model updates, monitoring). Key concerns: data leakage, concept drift, uncertainty quantification. Production tips: start simple (naive baselines), document assumptions, monitor forecast accuracy over time, have fallback strategies.
</div>

## Core Definitions

**Backtesting:** Historical simulation of how model would have performed.

**Walk-Forward Validation:** Expanding window validation mimicking production.

**Concept Drift:** When relationship between features and target changes over time.

**Model Retraining:** Updating model with new data periodically.

**Forecast Reconciliation:** Ensuring forecasts at different aggregation levels are consistent.

## Math and Derivations

### Rolling Origin Backtest

For origins $T_1, T_2, \ldots, T_m$ and horizon h:
$$\text{RMSE}(h) = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_{T_i+h} - \hat{y}_{T_i+h|T_i})^2}$$

### Forecast Bias

$$\text{Bias} = \frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)$$

- Positive bias: systematic under-prediction
- Negative bias: systematic over-prediction

### Tracking Signal (for monitoring)

$$TS_t = \frac{\sum_{i=1}^{t}e_i}{\text{MAD}}$$

If |TS| > 4, model may be biased and needs retraining.

### Prediction Interval Coverage

$$\text{Coverage} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}(y_t \in PI_t)$$

95% PI should have ~95% coverage; significantly less indicates miscalibration.

## Algorithm/Model Sketch

**Production Forecasting Pipeline:**

```python
def production_pipeline(data, config):
    """
    Complete forecasting pipeline for production.
    """
    # 1. Data validation
    validate_data(data)

    # 2. Feature engineering
    features = create_features(data)

    # 3. Train-test split (temporal)
    train, holdout = temporal_split(features, config['holdout_size'])

    # 4. Model selection via cross-validation
    best_model = None
    best_score = float('inf')

    for model_class in config['candidate_models']:
        score = time_series_cv(train, model_class, config['cv_folds'])
        if score < best_score:
            best_model = model_class
            best_score = score

    # 5. Final training on full training set
    model = best_model.fit(train)

    # 6. Holdout evaluation
    holdout_metrics = evaluate(model, holdout)

    # 7. Retrain on all data for deployment
    final_model = best_model.fit(features)

    # 8. Generate forecasts with intervals
    forecasts = final_model.forecast(config['horizon'])
    intervals = final_model.prediction_intervals(config['horizon'])

    return {
        'model': final_model,
        'forecasts': forecasts,
        'intervals': intervals,
        'metrics': holdout_metrics
    }
```

**Monitoring Dashboard Metrics:**

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| MAPE | < 10% | 10-20% | > 20%: investigate |
| Bias | ≈ 0 | |bias| > 1σ | |bias| > 2σ: retrain |
| PI Coverage | 90-100% | 80-90% | < 80%: recalibrate |
| Tracking Signal | |TS| < 4 | 4-6 | > 6: retrain |

## Common Pitfalls

1. **Random train-test split:** Causes data leakage. Always use temporal splits.

2. **Optimizing wrong metric:** Minimize business-relevant loss (e.g., asymmetric cost), not just RMSE.

3. **No baseline comparison:** Claim "model works" without comparing to naive/seasonal naive.

4. **Static model:** Not retraining as new data arrives. Monitor performance and retrain regularly.

5. **Ignoring prediction intervals:** Point forecasts without uncertainty mislead decision-makers.

6. **Overfitting to holdout:** If you tune on holdout multiple times, it becomes training data. Use nested CV.

## Mini Example

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def tracking_signal(errors):
    cumsum = np.cumsum(errors)
    mad = np.mean(np.abs(errors))
    return cumsum[-1] / mad if mad > 0 else 0

# Simulate production monitoring
np.random.seed(42)
n_periods = 12  # 12 months of monitoring

actuals = 100 + np.random.randn(n_periods) * 10
forecasts = actuals + np.random.randn(n_periods) * 5 + 2  # slight bias

errors = actuals - forecasts

# Calculate monitoring metrics
print("=== Forecast Monitoring Report ===\n")
print(f"MAE: {mean_absolute_error(actuals, forecasts):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actuals, forecasts)):.2f}")
print(f"MAPE: {mape(actuals, forecasts):.2f}%")
print(f"Bias: {np.mean(errors):.2f}")
print(f"Tracking Signal: {tracking_signal(errors):.2f}")

# PI coverage check (simulated 95% intervals)
pi_width = 1.96 * np.std(errors)
lower = forecasts - pi_width
upper = forecasts + pi_width
coverage = np.mean((actuals >= lower) & (actuals <= upper))
print(f"PI Coverage: {coverage*100:.1f}%")

# Alert check
print("\n=== Alerts ===")
if abs(np.mean(errors)) > 2 * np.std(errors):
    print("WARNING: Significant forecast bias detected!")
if abs(tracking_signal(errors)) > 4:
    print("WARNING: Tracking signal exceeds threshold - consider retraining")
if coverage < 0.80:
    print("WARNING: Prediction interval coverage too low")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why is random train-test split wrong for time series?</summary>

<div class="answer">
<strong>Answer:</strong> Random splits cause data leakage:
- Future observations appear in training set
- Past observations appear in test set
- Model "sees" future information during training

**Consequences:**
- Overoptimistic evaluation metrics
- Model fails in production where future isn't available
- Temporal patterns learned incorrectly

**Correct approach:**
```
Train: [1, ..., T]    Test: [T+1, ..., T+h]
```
Always train on past, test on future.

<div class="pitfall">
<strong>Common pitfall:</strong> Using sklearn's train_test_split() or KFold directly on time series. Use TimeSeriesSplit or manual temporal split.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> What is concept drift and how do you detect it?</summary>

<div class="answer">
<strong>Answer:</strong> Concept drift = the relationship between features and target changes over time.

**Types:**
- **Sudden:** Abrupt change (e.g., COVID impact)
- **Gradual:** Slow shift (e.g., customer behavior evolution)
- **Seasonal:** Recurring pattern changes
- **Recurring:** Oscillates between states

**Detection methods:**
1. **Performance monitoring:** Increasing error over time
2. **Statistical tests:** Compare recent vs historical distributions
3. **Control charts:** Track forecast errors, flag out-of-control
4. **Tracking signal:** Cumulative bias indicates drift

**Response:**
- Retrain on recent data
- Use adaptive models (exponential smoothing)
- Reduce lookback window
- Add regime indicators

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming model stays accurate forever. Schedule regular monitoring and retraining.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> How do you calculate and interpret the tracking signal?</summary>

<div class="answer">
<strong>Answer:</strong>
$$TS_t = \frac{\text{RSFE}_t}{\text{MAD}_t} = \frac{\sum_{i=1}^{t}e_i}{\frac{1}{t}\sum_{i=1}^{t}|e_i|}$$

**Interpretation:**
- TS ≈ 0: No systematic bias
- TS > 0: Systematic under-forecasting
- TS < 0: Systematic over-forecasting
- |TS| > 4: Likely significant bias (action needed)

**Why use it:**
- Normalizes by MAD for comparability
- Accumulates evidence over time
- Distinguishes random errors from systematic bias

**Update frequency:**
Check monthly or quarterly; daily TS is noisy.

<div class="pitfall">
<strong>Common pitfall:</strong> Reacting to every TS fluctuation. Wait for sustained signal (multiple periods with |TS| > 4) before retraining.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> What does it mean if your 95% prediction intervals have 75% coverage?</summary>

<div class="answer">
<strong>Answer:</strong> The model underestimates uncertainty—intervals are too narrow.

**Possible causes:**
1. **Model misspecification:** True residuals are larger than estimated
2. **Heavy tails:** Data has outliers not captured by normal assumption
3. **Non-constant variance:** Heteroskedasticity not modeled
4. **Missing patterns:** Unmodeled seasonality or trend adds variance

**Solutions:**
1. Use bootstrap prediction intervals (more robust)
2. Apply variance adjustment: multiply width by coverage correction factor
3. Model heteroskedasticity (GARCH) or use quantile regression
4. Improve base model to capture more patterns

**Adjustment formula:**
If coverage = 75% but target = 95%, scale factor ≈ $z_{0.975}/z_{0.875}$ = 1.96/1.15 ≈ 1.7

<div class="pitfall">
<strong>Common pitfall:</strong> Reporting narrow intervals to seem accurate. Stakeholders need truthful uncertainty; too-narrow intervals cause poor decisions.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> Your demand forecasting model works well in testing but production accuracy is much worse. What might be causing this?</summary>

<div class="answer">
<strong>Answer:</strong> Common causes of train-test vs production gap:

1. **Data leakage in testing:**
   - Features used information not available at forecast time
   - Train-test split not strictly temporal

2. **Feature availability:**
   - Features available historically but delayed in production
   - External data sources not updated in real-time

3. **Distribution shift:**
   - Test period was unusual/lucky
   - Production faces different conditions (seasonality, promotions)

4. **Data quality:**
   - Production data has errors/delays not in historical
   - Missing values handled differently

5. **Target leakage:**
   - Test evaluated at easy horizons; production needs longer

**Diagnosis:**
1. Verify no leakage in feature engineering
2. Compare feature distributions: test vs production
3. Backtest over multiple periods (not just one)
4. Monitor input data quality in production
5. Check if production horizon matches testing

<div class="pitfall">
<strong>Common pitfall:</strong> Testing on convenient period and assuming it generalizes. Always backtest across multiple train-test splits spanning different conditions.
</div>
</div>
</details>

## References

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapters 5, 12.
2. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *IJF*, 36(1), 54-74.
3. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1-37.
4. Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *IJF*, 32(3), 788-803.
