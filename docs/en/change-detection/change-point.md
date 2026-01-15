# Change-Point Detection

<div class="interview-summary">
<strong>Interview Summary:</strong> Change-point detection identifies times when statistical properties (mean, variance, trend) shift. Methods: CUSUM (cumulative sum), PELT (penalized exact linear time), Bayesian online detection. Key trade-off: sensitivity vs. false positives. Applications: process monitoring, regime detection, structural breaks in economics. For offline detection, use dynamic programming; for online, use sequential methods.
</div>

## Core Definitions

**Change-Point:** Time $\tau$ where distribution parameters change:
$$y_t \sim \begin{cases} F_1(\theta_1) & t < \tau \\ F_2(\theta_2) & t \geq \tau \end{cases}$$

**Types:**
- Mean shift: $\mu_1 \neq \mu_2$
- Variance change: $\sigma_1^2 \neq \sigma_2^2$
- Trend break: Slope changes
- Multiple change-points: $\tau_1 < \tau_2 < \cdots < \tau_k$

**Settings:**
- Offline: All data available, find all change-points
- Online: Sequential, detect changes as they occur

## Math and Derivations

### CUSUM (Cumulative Sum)

For mean detection with known parameters:
$$S_t = \sum_{i=1}^{t}(y_i - \mu_0)$$

Under H₀ (no change): $S_t$ fluctuates around 0
Under H₁ (mean shift at τ): $S_t$ trends away from 0 after τ

**Page's CUSUM:**
$$C_t^+ = \max(0, C_{t-1}^+ + y_t - \mu_0 - k)$$
$$C_t^- = \max(0, C_{t-1}^- - y_t + \mu_0 - k)$$

Signal change when $C_t^+ > h$ or $C_t^- > h$

Parameters: $k$ (allowance), $h$ (threshold)

### Binary Segmentation

Greedy algorithm for multiple change-points:

1. Test for one change-point in [1, T]
2. If found at τ₁, recursively search [1, τ₁) and [τ₁, T]
3. Continue until no more significant changes

Cost function (e.g., RSS):
$$C(y_{s:t}) = \sum_{i=s}^{t}(y_i - \bar{y}_{s:t})^2$$

### PELT (Pruned Exact Linear Time)

Optimal partitioning via dynamic programming:
$$F(t) = \min_{s < t}\{F(s) + C(y_{s+1:t}) + \beta\}$$

where β is penalty per change-point.

**Pruning:** Eliminate suboptimal segmentations to achieve O(n) complexity.

### Bayesian Online Change-Point Detection

Maintain probability distribution over run length $r_t$ (time since last change):
$$P(r_t | y_{1:t}) \propto P(y_t | r_t, y_{1:t-1}) P(r_t | r_{t-1})$$

Growth probability: $P(r_t = r_{t-1} + 1)$
Change probability: $P(r_t = 0)$

## Algorithm/Model Sketch

**Offline Detection (PELT):**

```python
def pelt(y, penalty, min_size=2):
    n = len(y)
    F = [0]  # F[t] = min cost for y[0:t]
    cp = [[]]  # change-points for optimal segmentation

    for t in range(1, n + 1):
        candidates = []
        for s in range(max(0, t - max_segments), t):
            if t - s >= min_size:
                cost = F[s] + segment_cost(y[s:t]) + penalty
                candidates.append((cost, s))

        best_cost, best_s = min(candidates)
        F.append(best_cost)
        cp.append(cp[best_s] + [best_s] if best_s > 0 else [])

    return cp[-1]
```

**Online Detection (CUSUM):**

```python
def cusum_online(y, mu0, k, h):
    n = len(y)
    C_plus, C_minus = 0, 0
    alarms = []

    for t in range(n):
        C_plus = max(0, C_plus + y[t] - mu0 - k)
        C_minus = max(0, C_minus - y[t] + mu0 - k)

        if C_plus > h or C_minus > h:
            alarms.append(t)
            C_plus, C_minus = 0, 0  # Reset

    return alarms
```

## Common Pitfalls

1. **Penalty selection**: Too small → over-segmentation; too large → miss changes. Use BIC-based penalties or cross-validation.

2. **Minimum segment length**: Very short segments are often noise. Enforce minimum size constraint.

3. **Multiple testing**: Testing many potential change-points inflates false positives. Adjust thresholds.

4. **Model misspecification**: Assuming wrong distribution (e.g., normal for heavy-tailed data) affects detection.

5. **Gradual vs. abrupt changes**: Most methods assume sudden shifts. Gradual changes may appear as multiple small changes.

6. **Online delay**: Online methods detect changes with delay. Trade-off between speed and accuracy.

## Mini Example

```python
import numpy as np
import ruptures as rpt

# Generate data with 2 change-points
np.random.seed(42)
n = 300

# Three segments with different means
y = np.concatenate([
    np.random.randn(100) + 0,      # mean 0
    np.random.randn(100) + 3,      # mean 3
    np.random.randn(100) + 1       # mean 1
])

# PELT detection
algo = rpt.Pelt(model="l2", min_size=10).fit(y)
change_points = algo.predict(pen=10)
print(f"Detected change-points: {change_points[:-1]}")
print(f"True change-points: [100, 200]")

# Binary segmentation
algo_binseg = rpt.Binseg(model="l2", min_size=10).fit(y)
change_points_bs = algo_binseg.predict(n_bkps=2)
print(f"BinSeg change-points: {change_points_bs[:-1]}")

# Bayesian approach (conceptual)
# Using online Bayesian detection
from scipy.stats import norm

def bocpd_simple(y, hazard=0.01, mu0=0, sigma=1):
    """Simplified Bayesian Online CPD."""
    n = len(y)
    R = np.zeros((n + 1, n + 1))  # Run length probabilities
    R[0, 0] = 1

    for t in range(1, n + 1):
        # Predictive probability under each run length
        predprob = norm.pdf(y[t-1], mu0, sigma)  # Simplified

        # Growth probability
        R[t, 1:t+1] = R[t-1, :t] * predprob * (1 - hazard)
        # Change probability
        R[t, 0] = np.sum(R[t-1, :t] * predprob * hazard)
        # Normalize
        R[t, :] /= R[t, :].sum()

    return R

R = bocpd_simple(y, hazard=0.01, mu0=1, sigma=1.5)
print(f"Max probability run length at end: {np.argmax(R[-1])}")
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the trade-off between online and offline change-point detection?</summary>

<div class="answer">
<strong>Answer:</strong>

**Online detection:**
- ✓ Real-time alerts
- ✓ No need for full data
- ✗ Detection delay (need evidence to accumulate)
- ✗ Can't revise past decisions
- Use case: Process monitoring, fraud detection

**Offline detection:**
- ✓ Uses all data for optimal segmentation
- ✓ Can find exact change locations
- ✓ More accurate (global optimization)
- ✗ Not real-time
- Use case: Historical analysis, model building

**Hybrid:** Some methods (e.g., Bayesian online) can be run offline for retrospective analysis while also providing online capability.

<div class="pitfall">
<strong>Common pitfall:</strong> Using online methods for offline analysis. If all data is available, use PELT or exact methods for better accuracy.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> How do you choose the penalty parameter in PELT or similar methods?</summary>

<div class="answer">
<strong>Answer:</strong> Several approaches:

1. **BIC-style penalty:** $\beta = \log(n) \times k$ where k = parameters per segment
   - Theoretically justified
   - Can be conservative

2. **Cross-validation:**
   - Hold out data, select penalty minimizing prediction error
   - Computationally intensive

3. **Elbow method:**
   - Plot cost vs. number of change-points
   - Select "elbow" where diminishing returns begin

4. **Domain knowledge:**
   - Expected number of changes
   - Cost of false positives vs. missed detections

**SIC/MBIC:** Modified BIC for change-point specific context.

<div class="pitfall">
<strong>Common pitfall:</strong> Using fixed penalty across different datasets. Optimal penalty depends on signal-to-noise ratio, segment lengths, and number of observations.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the expected value of CUSUM under H₀ (no change) and H₁ (mean shift).</summary>

<div class="answer">
<strong>Answer:</strong>

**Under H₀:** $y_t \sim N(\mu_0, \sigma^2)$
$$E[S_t] = E\left[\sum_{i=1}^{t}(y_i - \mu_0)\right] = \sum_{i=1}^{t}E[y_i - \mu_0] = 0$$

CUSUM fluctuates around 0 (random walk behavior).

**Under H₁:** Mean shifts to $\mu_1$ at time $\tau$
$$E[S_t] = \sum_{i=1}^{\tau-1}(\mu_0 - \mu_0) + \sum_{i=\tau}^{t}(\mu_1 - \mu_0) = (t - \tau + 1)(\mu_1 - \mu_0)$$

After change, CUSUM drifts linearly away from 0 at rate $(\mu_1 - \mu_0)$.

**Key insight:** The drift rate equals the mean shift magnitude, making CUSUM sensitive to sustained changes.

<div class="pitfall">
<strong>Common pitfall:</strong> CUSUM assumes known pre-change mean $\mu_0$. If estimated, use standardized CUSUM or adjust thresholds.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why does binary segmentation not guarantee finding the optimal solution?</summary>

<div class="answer">
<strong>Answer:</strong> Binary segmentation is greedy—it makes locally optimal choices without considering global structure.

**Problem scenario:**
Consider data with two close change-points:
- Segment 1: mean 0, length 80
- Segment 2: mean 2, length 20
- Segment 3: mean 0, length 100

Binary segmentation first finds the "best" single split, which might be around position 100 (between segments 2 and 3). Then it searches [0,100] and [100,200] separately.

But the small segment 2 might be better detected by finding BOTH change-points (80 and 100) together.

**Solution:** Use exact methods (PELT, optimal partitioning) that consider all possible segmentations.

<div class="pitfall">
<strong>Common pitfall:</strong> Assuming binary segmentation is "good enough." For complex signals with varying segment sizes, exact methods can be significantly better.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You're monitoring server response times. You detect a "change-point" but it turns out to be a single outlier. How do you prevent false alarms from outliers?</summary>

<div class="answer">
<strong>Answer:</strong> Several strategies:

1. **Robust cost functions:**
   - Use L1 (absolute) instead of L2 (squared)
   - Huber loss
   - Median-based detection

2. **Minimum segment length:**
   - Require at least k observations in each segment
   - Single outliers can't form segments

3. **Pre-filtering:**
   - Apply median filter or outlier removal first
   - Then detect changes

4. **Multi-scale detection:**
   - Detect at multiple resolutions
   - True changes appear at all scales; outliers don't

5. **Confirmation period:**
   - Don't alarm on first deviation
   - Require sustained change (e.g., CUSUM with appropriate k)

6. **Model-based:**
   - Use models that explicitly include outlier component
   - Separate outliers from level shifts

<div class="pitfall">
<strong>Common pitfall:</strong> Using squared error cost with heavy-tailed data. Single large values dominate and trigger false change-points.
</div>
</div>
</details>

## References

1. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
2. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA*, 107(500), 1590-1598.
3. Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
4. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.
