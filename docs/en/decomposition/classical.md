# Classical Decomposition

<div class="interview-summary">
<strong>Interview Summary:</strong> Classical decomposition separates a series into trend, seasonal, and irregular components using moving averages. Additive: $y = T + S + I$. Multiplicative: $y = T \times S \times I$. Simple but assumes constant seasonal pattern and is sensitive to outliers. Trend extracted via centered moving average; seasonal via period averages.
</div>

## Core Definitions

**Additive Model:**
$$y_t = T_t + S_t + I_t$$

Used when seasonal variation is constant regardless of level.

**Multiplicative Model:**
$$y_t = T_t \times S_t \times I_t$$

Used when seasonal variation scales with level.

**Components:**
- $T_t$: Trend-cycle (smooth underlying level)
- $S_t$: Seasonal (periodic pattern repeating every $m$ periods)
- $I_t$: Irregular/residual (random noise)

**Seasonal Indices:**
- Additive: $S_t$ values sum to 0 over one period
- Multiplicative: $S_t$ values average to 1 over one period

## Math and Derivations

### Trend Extraction via Moving Average

**Centered Moving Average (CMA):**

For odd $m$ (e.g., $m=7$):
$$T_t = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} y_{t+j}$$

For even $m$ (e.g., $m=12$):
$$T_t = \frac{1}{2m}\left(y_{t-m/2} + 2\sum_{j=-(m/2-1)}^{m/2-1} y_{t+j} + y_{t+m/2}\right)$$

This is a $2 \times m$-MA: first $m$-MA, then 2-MA to center.

### Seasonal Index Calculation

**Additive:**
1. Detrend: $y_t - T_t$
2. Average detrended values for each season: $\bar{S}_s = \frac{1}{k}\sum_{j} (y_{s+jm} - T_{s+jm})$
3. Normalize: $S_s = \bar{S}_s - \frac{1}{m}\sum_{s=1}^{m}\bar{S}_s$

**Multiplicative:**
1. Detrend: $y_t / T_t$
2. Average ratios for each season: $\bar{S}_s = \frac{1}{k}\sum_{j} \frac{y_{s+jm}}{T_{s+jm}}$
3. Normalize: $S_s = \bar{S}_s \times \frac{m}{\sum_{s=1}^{m}\bar{S}_s}$

### Properties of Moving Average

The $m$-point moving average:
- Removes seasonality of period $m$ (averages over full cycle)
- Smooths high-frequency noise
- Introduces lag of $(m-1)/2$ periods
- Loses $(m-1)/2$ observations at each end

**Frequency response:** MA is a low-pass filter that attenuates frequencies $\geq 1/m$.

## Algorithm/Model Sketch

**Classical Decomposition Algorithm:**

```
Input: y[1:n], seasonal period m, type (additive/multiplicative)
Output: Trend T, Seasonal S, Irregular I

1. TREND EXTRACTION
   - Compute centered moving average of y
   - T[t] = CMA(y, m) for t = m/2+1 to n-m/2
   - End points: use extrapolation or leave missing

2. DETREND
   - Additive: D[t] = y[t] - T[t]
   - Multiplicative: D[t] = y[t] / T[t]

3. SEASONAL INDICES
   - Group D[t] by season (1 to m)
   - Average each group: S_raw[s] = mean(D[s], D[s+m], D[s+2m],...)
   - Normalize:
     - Additive: S[s] = S_raw[s] - mean(S_raw)
     - Multiplicative: S[s] = S_raw[s] × m / sum(S_raw)

4. SEASONAL COMPONENT
   - S[t] = S[t mod m] (replicate indices across series)

5. IRREGULAR
   - Additive: I[t] = y[t] - T[t] - S[t]
   - Multiplicative: I[t] = y[t] / (T[t] × S[t])

Return T, S, I
```

## Common Pitfalls

1. **Fixed seasonal pattern**: Classical decomposition assumes the same seasonal pattern throughout. Doesn't adapt to evolving seasonality.

2. **Outlier sensitivity**: One outlier affects trend (via MA) and seasonal indices. No robust fitting.

3. **End-point loss**: Lose $m/2$ observations at each end. Problematic for short series.

4. **Wrong model type**: Using additive when data is multiplicative (or vice versa) gives poor decomposition.

5. **Calendar effects**: Doesn't handle trading days, Easter, etc. These appear in irregular component.

6. **Non-integer period**: Requires integer $m$. For 365.25 days/year, need alternative methods.

## Mini Example

```python
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate multiplicative seasonal data
np.random.seed(42)
n = 48  # 4 years monthly
t = np.arange(n)
trend = 100 + 2 * t
seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
y = trend * seasonal_mult * (1 + 0.05 * np.random.randn(n))

# Classical decomposition (multiplicative)
result = seasonal_decompose(y, model='multiplicative', period=12)

print("Seasonal indices (should repeat):")
print(np.round(result.seasonal[:12], 3))

print("\nTrend (first and last available):")
print(f"  First: {result.trend[~np.isnan(result.trend)][0]:.1f}")
print(f"  Last: {result.trend[~np.isnan(result.trend)][-1]:.1f}")

# Compare additive (wrong model for this data)
result_add = seasonal_decompose(y, model='additive', period=12)
print("\nCompare residual std:")
print(f"  Multiplicative: {np.nanstd(result.resid):.3f}")
print(f"  Additive: {np.nanstd(result_add.resid):.3f}")
# Multiplicative should have smaller residual std
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> When should you use additive vs. multiplicative decomposition?</summary>

<div class="answer">
<strong>Answer:</strong>

**Additive** when:
- Seasonal fluctuations are roughly constant in absolute terms
- Low and high values have similar seasonal swings
- Example: Temperature (±10°F regardless of base temp)

**Multiplicative** when:
- Seasonal fluctuations are proportional to level
- Percentage variation is constant
- Example: Retail sales (December is 20% above average regardless of total sales)

**Decision test:**
1. Plot series — do seasonal swings grow with level?
2. Compute: std(seasonal) / mean(level) for different periods
   - If ratio is constant → multiplicative
   - If std(seasonal) is constant → additive

<div class="pitfall">
<strong>Common pitfall:</strong> Defaulting to additive. Many economic/business series are multiplicative because growth compounds.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Why does the moving average remove seasonality of period m?</summary>

<div class="answer">
<strong>Answer:</strong> An $m$-point moving average includes exactly one complete seasonal cycle. Since seasonal components sum to zero (additive) or average to one (multiplicative) over a cycle, they cancel out.

**Mathematical explanation (additive):**
$$\text{MA}_t = \frac{1}{m}\sum_{j=0}^{m-1}(T_{t+j} + S_{t+j} + I_{t+j})$$

If trend is locally constant and $\sum_{j=0}^{m-1}S_{t+j} = 0$:
$$\text{MA}_t \approx T_t + \frac{1}{m}\sum_{j=0}^{m-1}I_{t+j}$$

The seasonal cancels; only trend and smoothed noise remain.

<div class="pitfall">
<strong>Common pitfall:</strong> Using wrong MA order. If true period is 12 but you use MA(6), seasonality won't be fully removed.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> For a 12-point centered moving average of monthly data, how many observations are lost at each end?</summary>

<div class="answer">
<strong>Answer:</strong> 6 observations at each end (12 total).

**Derivation:**
For even $m=12$, the 2×12-MA formula at time $t$ uses:
$$\frac{1}{24}(y_{t-6} + 2y_{t-5} + \cdots + 2y_{t+5} + y_{t+6})$$

This requires observations from $t-6$ to $t+6$.

At the start: Can only compute for $t \geq 7$ (need $y_1, \ldots, y_{13}$)
At the end: Can only compute for $t \leq n-6$ (need data through $y_{n}$)

So positions 1-6 and (n-5)-n are missing → 12 total missing values.

<div class="pitfall">
<strong>Common pitfall:</strong> For short series (say, 2 years = 24 points), losing 12 points means half the data has no trend estimate. Consider STL or parametric trend instead.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why must seasonal indices be normalized, and what constraint do they satisfy?</summary>

<div class="answer">
<strong>Answer:</strong>

**Why normalize:**
Without normalization, average of raw seasonal indices might not be zero (additive) or one (multiplicative), causing systematic bias in trend or irregular.

**Constraints:**

Additive: $\sum_{s=1}^{m} S_s = 0$
- Ensures seasonality doesn't shift the overall level
- Positive seasons offset by negative seasons

Multiplicative: $\sum_{s=1}^{m} S_s = m$ (equivalently, average = 1)
- Ensures seasonal factors don't inflate/deflate overall level
- Factors > 1 offset by factors < 1

**Normalization formulas:**
- Additive: $S_s^{new} = S_s^{raw} - \bar{S}^{raw}$
- Multiplicative: $S_s^{new} = S_s^{raw} \times m / \sum S^{raw}$

<div class="pitfall">
<strong>Common pitfall:</strong> Forgetting normalization leads to decomposition where $T + S + I \neq y$ due to bias in seasonal.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You decompose monthly sales data and find the irregular component has strong autocorrelation at lag 1. What does this indicate and how would you address it?</summary>

<div class="answer">
<strong>Answer:</strong> Lag-1 autocorrelation in irregular indicates:
1. The model hasn't captured all systematic patterns
2. Short-term dynamics exist beyond trend and seasonality
3. Possibly: month-to-month momentum not in seasonal pattern

**How to address:**

1. **Model the irregular:** Fit ARIMA to irregular component
   - If AR(1), incorporate into forecasting
   - STL+ARIMA or similar pipeline

2. **Use better decomposition:** STL can adapt to changing patterns that classical misses

3. **Consider SARIMA directly:** Handles trend, seasonality, AND autocorrelation in one model

4. **Check for calendar effects:** Trading days, holidays may create autocorrelation

5. **Use ETS:** State space models can capture autocorrelated errors

<div class="pitfall">
<strong>Common pitfall:</strong> Ignoring irregular autocorrelation in forecasting. This underestimates forecast uncertainty and may bias predictions.
</div>
</div>
</details>

## References

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts. Chapter 3.
2. Makridakis, S., Wheelwright, S. C., & Hyndman, R. J. (1998). *Forecasting: Methods and Applications*. Wiley. Chapter 4.
3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 1.
4. Census Bureau. (2017). X-13ARIMA-SEATS Reference Manual. U.S. Census Bureau.
