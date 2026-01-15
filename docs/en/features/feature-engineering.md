# Time Series Feature Engineering

<div class="interview-summary">
<strong>Interview Summary:</strong> Feature engineering transforms raw time series into ML-ready inputs. Key features: lags, rolling statistics, date/time features, Fourier terms for seasonality. Scaling: standardize or min-max, but fit only on training data. Handle missing data via interpolation or indicators. Avoid leakage: never use future information in features.
</div>

## Core Definitions

**Lag Features:** Past values as predictors
$$X_{lag_k} = y_{t-k}$$

**Rolling Features:** Statistics over windows
$$X_{roll\_mean\_k} = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}$$

**Date Features:** Extracted from timestamp
- Hour, day of week, month, quarter
- Is_weekend, is_holiday
- Days since event

**Fourier Features:** Seasonal patterns as sine/cosine
$$X_{sin_k} = \sin\left(\frac{2\pi k t}{m}\right), \quad X_{cos_k} = \cos\left(\frac{2\pi k t}{m}\right)$$

## Math and Derivations

### Fourier Terms for Seasonality

For seasonal period m, capture pattern with K harmonics:
$$s(t) = \sum_{k=1}^{K}\left[\alpha_k\sin\left(\frac{2\pi kt}{m}\right) + \beta_k\cos\left(\frac{2\pi kt}{m}\right)\right]$$

**K selection:**
- K = m/2: captures all seasonal frequencies
- K = 2-4: often sufficient for smooth patterns
- Use AIC to select optimal K

**Why Fourier:**
- Handles non-integer and long seasonal periods
- Works with any ML model
- Parsimonious: 2K features vs m dummy variables

### Rolling Statistics

**Rolling mean (simple moving average):**
$$\bar{y}_t^{(w)} = \frac{1}{w}\sum_{i=0}^{w-1}y_{t-i}$$

**Rolling standard deviation:**
$$s_t^{(w)} = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1}(y_{t-i} - \bar{y}_t^{(w)})^2}$$

**Exponential moving average:**
$$\text{EMA}_t = \alpha y_t + (1-\alpha)\text{EMA}_{t-1}$$

### Scaling Methods

**Standardization (z-score):**
$$y_{scaled} = \frac{y - \mu_{train}}{\sigma_{train}}$$

**Min-max scaling:**
$$y_{scaled} = \frac{y - \min_{train}}{\max_{train} - \min_{train}}$$

**Robust scaling:**
$$y_{scaled} = \frac{y - \text{median}_{train}}{\text{IQR}_{train}}$$

**Critical:** Always fit scaler on training data only!

## Algorithm/Model Sketch

**Feature Engineering Pipeline:**

```python
def create_features(df, target_col='y', lags=[1,2,3,7],
                   rolling_windows=[7,14,30]):
    features = df.copy()

    # Lag features
    for lag in lags:
        features[f'lag_{lag}'] = features[target_col].shift(lag)

    # Rolling features
    for w in rolling_windows:
        features[f'roll_mean_{w}'] = features[target_col].shift(1).rolling(w).mean()
        features[f'roll_std_{w}'] = features[target_col].shift(1).rolling(w).std()
        features[f'roll_min_{w}'] = features[target_col].shift(1).rolling(w).min()
        features[f'roll_max_{w}'] = features[target_col].shift(1).rolling(w).max()

    # Date features (if datetime index)
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    features['is_weekend'] = features.index.dayofweek >= 5

    # Fourier features for annual seasonality
    day_of_year = features.index.dayofyear
    for k in range(1, 4):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    return features.dropna()
```

**Train-Test Split for Time Series:**
```python
# WRONG: random split
X_train, X_test = train_test_split(X)  # Data leakage!

# RIGHT: temporal split
train_end = int(len(X) * 0.8)
X_train, X_test = X[:train_end], X[train_end:]
```

## Common Pitfalls

1. **Using future information:** Lag features must use shift(k) where k ≥ 1. shift(0) = leakage.

2. **Scaling on full data:** Fit scaler on training data only. Otherwise test data statistics leak into training.

3. **Rolling windows including current value:** Rolling mean should be `.shift(1).rolling(w)`, not `.rolling(w)`.

4. **Missing values from lags:** First k observations have NaN after creating lag_k. Drop or impute.

5. **Too many features:** With many lags and rolling windows, dimensionality explodes. Use feature selection.

6. **Non-stationarity in features:** If target is non-stationary, lag features inherit it. Consider differencing.

## Mini Example

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
n = 365
dates = pd.date_range('2023-01-01', periods=n, freq='D')
trend = np.arange(n) * 0.1
seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 2
y = trend + seasonal + noise

df = pd.DataFrame({'y': y}, index=dates)

# Create features
def make_features(df):
    features = df.copy()
    # Lags
    for lag in [1, 7, 14, 28]:
        features[f'lag_{lag}'] = features['y'].shift(lag)
    # Rolling
    features['roll_mean_7'] = features['y'].shift(1).rolling(7).mean()
    features['roll_std_7'] = features['y'].shift(1).rolling(7).std()
    # Calendar
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    # Fourier (annual)
    doy = features.index.dayofyear
    features['sin_annual'] = np.sin(2 * np.pi * doy / 365.25)
    features['cos_annual'] = np.cos(2 * np.pi * doy / 365.25)
    return features.dropna()

features = make_features(df)

# Train-test split (temporal)
train_size = 300
train = features[:train_size]
test = features[train_size:]

X_train = train.drop('y', axis=1)
y_train = train['y']
X_test = test.drop('y', axis=1)
y_test = test['y']

# Fit model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# Feature importance
importance = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop features:")
print(importance.sort_values(ascending=False).head())
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Why must you fit the scaler only on training data?</summary>

<div class="answer">
<strong>Answer:</strong> Fitting on full data causes data leakage—test data statistics influence training.

**Problem:**
```python
scaler.fit(X)  # Uses test data statistics
X_train_scaled = scaler.transform(X_train)  # Training influenced by test
```

The model "knows" about test data range/distribution during training, giving optimistic evaluation.

**Correct approach:**
```python
scaler.fit(X_train)  # Only training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply same transformation
```

**Real-world analogy:** In production, you don't have future data to compute statistics.

<div class="pitfall">
<strong>Common pitfall:</strong> Using sklearn pipelines incorrectly. Always split BEFORE creating pipeline, or use TimeSeriesSplit in cross-validation.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> When would you use Fourier features instead of dummy variables for seasonality?</summary>

<div class="answer">
<strong>Answer:</strong>

**Use Fourier when:**
1. **Long periods:** Weekly seasonality = 7 dummies; annual = 365 dummies vs 2-6 Fourier
2. **Non-integer periods:** 365.25 days/year can't be captured with dummies
3. **Smooth patterns:** Seasonality follows sinusoidal shape
4. **Linear models:** Fourier terms capture cycles naturally

**Use dummies when:**
1. **Short periods:** Day of week (7 levels) is manageable
2. **Sharp patterns:** "Monday effect" is discrete, not smooth
3. **Interpretability:** Coefficients directly show day effects
4. **Non-sinusoidal:** Pattern doesn't fit sine/cosine shape

**Hybrid:** Can use both—Fourier for smooth annual, dummies for weekly.

<div class="pitfall">
<strong>Common pitfall:</strong> Using 52 dummies for weekly seasonality in daily data. Fourier with K=2-4 is more efficient and generalizes better.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Explain why shift(1).rolling(w).mean() is correct but rolling(w).mean() causes leakage.</summary>

<div class="answer">
<strong>Answer:</strong>

**Without shift:**
```python
rolling_mean[t] = mean(y[t-w+1], ..., y[t])  # Includes y[t]!
```
When predicting y[t], using rolling_mean[t] includes y[t] itself → leakage.

**With shift:**
```python
rolling_mean[t] = mean(y[t-w], ..., y[t-1])  # Excludes y[t]
```
Only uses past values → no leakage.

**Mathematical notation:**
- Wrong: $\bar{y}_t = \frac{1}{w}\sum_{i=0}^{w-1}y_{t-i}$ includes $y_t$
- Correct: $\bar{y}_{t-1} = \frac{1}{w}\sum_{i=1}^{w}y_{t-i}$ excludes $y_t$

The shift moves the window back by one time step.

<div class="pitfall">
<strong>Common pitfall:</strong> Pandas rolling default includes current value. Always add .shift(1) before .rolling() for features.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How many Fourier terms (K) do you need to fully represent seasonality of period m?</summary>

<div class="answer">
<strong>Answer:</strong> K = m/2 terms fully represent any periodic pattern of period m.

**Explanation:**
By Fourier's theorem, any periodic function can be represented as:
$$f(t) = \sum_{k=1}^{\infty}\left[a_k\sin\left(\frac{2\pi kt}{m}\right) + b_k\cos\left(\frac{2\pi kt}{m}\right)\right]$$

For discrete data with period m, frequencies above k = m/2 alias to lower frequencies (Nyquist).

**Practical:**
- K = m/2: Full representation (2K = m parameters, same as dummies)
- K = 2-4: Often sufficient; smooth patterns don't need high harmonics
- Use AIC/BIC: Add terms until no improvement

**Example:** Annual seasonality in daily data
- Full: K = 365/2 ≈ 182 (overkill)
- Typical: K = 3-5 (6-10 parameters)

<div class="pitfall">
<strong>Common pitfall:</strong> Using K = m/2 when K = 3 suffices. Extra terms add noise and reduce interpretability.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> Your lag features have 30% missing values at the start due to the lag window. How do you handle this?</summary>

<div class="answer">
<strong>Answer:</strong> Several options:

1. **Drop rows (simplest):**
   ```python
   features = features.dropna()
   ```
   - Lose initial observations
   - OK if plenty of data

2. **Fill with first available:**
   ```python
   features = features.fillna(method='bfill')
   ```
   - Uses earliest available value
   - Slight bias but preserves data

3. **Use target mean/median:**
   ```python
   features['lag_7'] = features['lag_7'].fillna(features['y'].mean())
   ```
   - Neutral imputation
   - Works for tree models

4. **Missing indicator:**
   ```python
   features['lag_7_missing'] = features['lag_7'].isna().astype(int)
   features['lag_7'] = features['lag_7'].fillna(0)
   ```
   - Model learns to handle missing
   - Most flexible

5. **Shorter warmup lags:**
   - Use lag_1 at start, add longer lags as available
   - Complex but maximizes data

<div class="pitfall">
<strong>Common pitfall:</strong> Dropping 30% of data when you have limited observations. Try imputation first; validate on holdout to check impact.
</div>
</div>
</details>

## References

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 7.
2. Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests. *Neurocomputing*, 307, 72-77.
3. Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework for automated time-series phenotyping. *Journal of Open Research Software*, 5(1).
4. Brownlee, J. (2018). *Deep Learning for Time Series Forecasting*. Machine Learning Mastery.
