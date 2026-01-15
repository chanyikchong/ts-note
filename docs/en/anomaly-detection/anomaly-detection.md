# Anomaly Detection

<div class="interview-summary">
<strong>Interview Summary:</strong> Anomaly detection identifies unusual observations—point anomalies (single outliers), collective anomalies (unusual sequences), or contextual anomalies (unusual given context). Methods: statistical (z-score, IQR), model-based (forecast residuals), distance-based (LOF), and ML (isolation forest, autoencoders). Key challenge: defining "normal" and choosing threshold.
</div>

## Core Definitions

**Anomaly Types:**
- **Point anomaly:** Single unusual value (e.g., spike)
- **Collective anomaly:** Sequence that's unusual as a group
- **Contextual anomaly:** Normal value in wrong context (e.g., high AC usage in winter)

**Detection Settings:**
- **Supervised:** Labeled normal/anomaly data
- **Semi-supervised:** Only normal data for training
- **Unsupervised:** No labels, detect statistical outliers

## Math and Derivations

### Statistical Methods

**Z-score:**
$$z_t = \frac{y_t - \bar{y}}{s}$$

Anomaly if $|z_t| > 3$ (or chosen threshold)

**Modified Z-score (robust):**
$$M_t = \frac{0.6745(y_t - \text{median})}{\text{MAD}}$$

where MAD = median(|y - median(y)|)

**IQR Method:**
$$\text{Anomaly if } y_t < Q_1 - 1.5\times IQR \text{ or } y_t > Q_3 + 1.5\times IQR$$

### Model-Based Detection

Fit time series model, flag large residuals:
$$e_t = y_t - \hat{y}_{t|t-1}$$

Anomaly if $|e_t| > k \times \hat{\sigma}$ (typically k = 3)

**Advantages:**
- Accounts for trend and seasonality
- Adapts to changing patterns
- More sensitive to contextual anomalies

### Isolation Forest

Anomalies are easier to isolate (require fewer splits).

**Algorithm:**
1. Build random trees by random splits
2. Anomaly score = average path length to isolate point
3. Short path → anomaly

**Score:**
$$s(x, n) = 2^{-E[h(x)]/c(n)}$$

where h(x) = path length, c(n) = average path length in random tree.

### Local Outlier Factor (LOF)

Compares local density to neighbors' density:
$$LOF(x) = \frac{\sum_{o \in N_k(x)} \frac{lrd(o)}{lrd(x)}}{|N_k(x)|}$$

LOF >> 1 → anomaly (lower density than neighbors)

## Algorithm/Model Sketch

**Time Series Anomaly Detection Pipeline:**

```python
def detect_anomalies(y, method='model', threshold=3):
    if method == 'zscore':
        z = (y - np.mean(y)) / np.std(y)
        return np.abs(z) > threshold

    elif method == 'model':
        # Fit model and get residuals
        model = fit_model(y)
        residuals = y - model.fittedvalues
        sigma = np.std(residuals)
        return np.abs(residuals) > threshold * sigma

    elif method == 'rolling':
        # Rolling window approach
        window = 30
        rolling_mean = y.rolling(window).mean()
        rolling_std = y.rolling(window).std()
        z = (y - rolling_mean) / rolling_std
        return np.abs(z) > threshold
```

**Threshold Selection:**
- Fixed (z > 3): Simple but may not fit data
- Percentile (top 1%): Adapts to distribution
- Domain-specific: Based on cost of false positives/negatives
- Extreme Value Theory: For tail events

## Common Pitfalls

1. **Masking:** One anomaly affects mean/std, hiding others. Use robust statistics or median-based methods.

2. **Swamping:** Normal points flagged due to anomaly influence. Sequential cleaning or robust fitting helps.

3. **Non-stationarity:** Using global statistics when local context matters. Use rolling windows or model residuals.

4. **Wrong threshold:** Fixed threshold may be too sensitive or too conservative. Tune based on validation data.

5. **Ignoring seasonality:** Saturday sales ≠ weekday anomaly. Model seasonal patterns first.

6. **Collective anomalies:** Point methods miss unusual sequences. Use sequence-aware methods.

## Mini Example

```python
import numpy as np
from scipy import stats

# Generate data with anomalies
np.random.seed(42)
n = 200
y = np.sin(2 * np.pi * np.arange(n) / 50) + np.random.randn(n) * 0.3

# Insert anomalies
anomaly_idx = [50, 100, 150]
y[anomaly_idx[0]] += 5   # Positive spike
y[anomaly_idx[1]] -= 4   # Negative spike
y[anomaly_idx[2]] += 3   # Smaller spike

# Method 1: Simple z-score
z_scores = np.abs(stats.zscore(y))
detected_zscore = np.where(z_scores > 3)[0]
print(f"Z-score detected: {detected_zscore}")

# Method 2: Rolling z-score
window = 20
rolling_mean = np.convolve(y, np.ones(window)/window, mode='same')
rolling_std = np.array([np.std(y[max(0,i-window):i+1]) for i in range(n)])
rolling_z = np.abs((y - rolling_mean) / rolling_std)
detected_rolling = np.where(rolling_z > 3)[0]
print(f"Rolling z-score detected: {detected_rolling[:10]}...")  # May have more

# Method 3: IQR
Q1, Q3 = np.percentile(y, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
detected_iqr = np.where((y < lower) | (y > upper))[0]
print(f"IQR detected: {detected_iqr}")

# Method 4: Model-based (simple AR)
from statsmodels.tsa.ar_model import AutoReg
model = AutoReg(y, lags=5).fit()
residuals = model.resid
res_z = np.abs(stats.zscore(residuals))
detected_model = np.where(res_z > 3)[0] + 5  # Adjust for lag offset
print(f"Model-based detected: {detected_model}")

print(f"\nTrue anomalies: {anomaly_idx}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the difference between a point anomaly and a contextual anomaly?</summary>

<div class="answer">
<strong>Answer:</strong>

**Point anomaly:** Value is unusual regardless of context.
- Example: Temperature reading of 500°F
- Detected by: Global statistics, simple thresholds

**Contextual anomaly:** Value is normal in some contexts, unusual in current context.
- Example: 80°F temperature is normal in summer, anomalous in winter
- Detected by: Model residuals, conditional distributions

**Why distinction matters:**
- Point methods (z-score) miss contextual anomalies
- Need context-aware methods for seasonal/temporal patterns
- False positives if using wrong method

<div class="pitfall">
<strong>Common pitfall:</strong> Using simple z-score on seasonal data. A December value might be normal for December but flagged as anomaly compared to annual mean.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Why is the median absolute deviation (MAD) preferred over standard deviation for anomaly detection?</summary>

<div class="answer">
<strong>Answer:</strong> MAD is robust to outliers; standard deviation is not.

**Problem with std:**
Single large outlier → inflates std → makes all z-scores smaller → outlier "masks" itself and others

**MAD robustness:**
$$\text{MAD} = \text{median}(|y_i - \text{median}(y)|)$$

- Median is not affected by extreme values
- 50% of data must be outliers to significantly affect MAD
- Breaking point: 50% vs ~0% for std

**Scale factor:**
For normal data: $\sigma \approx 1.4826 \times \text{MAD}$

Use modified z-score: $M = \frac{0.6745(y - \text{median})}{\text{MAD}}$

<div class="pitfall">
<strong>Common pitfall:</strong> Using std-based z-scores when data may contain multiple outliers. Masking effect hides true anomalies.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the modified z-score formula using MAD.</summary>

<div class="answer">
<strong>Answer:</strong> The modified z-score normalizes by MAD instead of std:

$$M_i = \frac{y_i - \tilde{y}}{\text{MAD}}$$

where $\tilde{y}$ = median(y).

**For comparison with standard z-score:**
For normal distribution: $E[\text{MAD}] = \Phi^{-1}(0.75) \times \sigma \approx 0.6745\sigma$

So: $\sigma \approx \frac{\text{MAD}}{0.6745}$

**Modified z-score (scaled):**
$$M_i = \frac{0.6745(y_i - \tilde{y})}{\text{MAD}}$$

Now M has same scale as standard z-score under normality.
Threshold M > 3.5 often used (equivalent to |z| > 3).

<div class="pitfall">
<strong>Common pitfall:</strong> Using threshold 3 for modified z-score without scaling factor. The raw MAD-based score has different scale than standard z.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How does Isolation Forest detect anomalies without computing distances?</summary>

<div class="answer">
<strong>Answer:</strong> Isolation Forest uses tree-based isolation:

**Key insight:** Anomalies are few and different → easier to isolate (separate from rest).

**Algorithm:**
1. Randomly select feature and split value
2. Recursively partition data
3. Anomaly score = average path length to isolate point

**Why anomalies have short paths:**
- Normal points: surrounded by similar points, need many splits
- Anomalies: isolated, few splits separate them

**Score formula:**
$$s(x,n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

- $E[h(x)]$ = expected path length for x
- $c(n)$ = average path length in binary search tree of n points
- $s \to 1$: anomaly; $s \to 0.5$: normal; $s \to 0$: very normal

<div class="pitfall">
<strong>Common pitfall:</strong> Isolation Forest assumes anomalies are isolated. Clustered anomalies (collective) may be missed.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You deploy an anomaly detection system and it triggers 50 alerts per day. After investigation, 45 are false positives. How do you improve?</summary>

<div class="answer">
<strong>Answer:</strong> 10% precision is problematic. Approaches:

1. **Raise threshold:**
   - Increase z-score cutoff from 3 to 4
   - Reduces false positives but may miss true anomalies

2. **Add context:**
   - Use time-of-day, day-of-week features
   - Model seasonal patterns
   - Contextual anomalies only within context

3. **Ensemble methods:**
   - Combine multiple detectors
   - Flag only if majority agree

4. **Learn from feedback:**
   - Label false positives as normal
   - Retrain semi-supervised model

5. **Two-stage detection:**
   - First stage: sensitive (catch all anomalies)
   - Second stage: verify (filter false positives)

6. **Domain rules:**
   - Add business logic filters
   - Known patterns that aren't anomalies

**Metrics to track:** Precision, recall, F1-score at different thresholds.

<div class="pitfall">
<strong>Common pitfall:</strong> Optimizing only for catching anomalies (recall). High false positive rate leads to alert fatigue and ignored warnings.
</div>
</div>
</details>

## References

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*, 413-422.
3. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *SIGMOD*, 93-104.
4. Hochenbaum, J., Vallis, O. S., & Kejariwal, A. (2017). Automatic anomaly detection in the cloud via statistical learning. *arXiv:1704.07706*.
