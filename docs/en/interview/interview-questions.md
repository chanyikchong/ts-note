# Common Interview Questions

<div class="interview-summary">
<strong>Interview Summary:</strong> This section compiles frequently asked time series interview questions with concise answers. Topics span fundamentals (stationarity, ACF/PACF), model selection (ARIMA vs. ETS vs. ML), practical challenges (seasonality, missing data, forecasting strategy), and advanced topics (state space, deep learning). Use this as a quick review before interviews.
</div>

## Fundamentals

### Q1: What is stationarity and why does it matter?

**Answer:** A stationary process has constant mean, constant variance, and autocovariance that depends only on lag (not time). It matters because:
- Most classical models (ARIMA, VAR) assume stationarity
- Non-stationary series have unpredictable long-term behavior
- Statistical tests and inference require stationarity

**Test for it:** ADF (null: unit root), KPSS (null: stationary). Use both for robust conclusions.

**Make it stationary:** Differencing (stochastic trend), detrending (deterministic trend), log transform (changing variance).

### Q2: Explain the difference between ACF and PACF.

**Answer:**
- **ACF:** Total correlation between $y_t$ and $y_{t-k}$, including indirect effects through intermediate lags
- **PACF:** Direct correlation between $y_t$ and $y_{t-k}$ after removing effects of $y_{t-1}, \ldots, y_{t-k+1}$

**Use for identification:**
- ACF cuts off at lag q → MA(q)
- PACF cuts off at lag p → AR(p)
- Both tail off → ARMA

### Q3: How do you handle seasonality?

**Answer:** Multiple approaches:
1. **Seasonal differencing:** $(1-L^s)y_t$ removes seasonal unit root
2. **SARIMA:** Explicit seasonal AR/MA terms
3. **Seasonal decomposition:** STL separates trend, seasonal, remainder
4. **Fourier terms:** Sin/cos features at seasonal frequencies
5. **Dummy variables:** For short seasonal periods

## Model Selection

### Q4: When would you choose ARIMA over exponential smoothing (ETS)?

**Answer:**
- **ARIMA:** When series has complex autocorrelation structure, when you need to include external regressors (ARIMAX), when interpretability of AR/MA structure matters
- **ETS:** When you want automatic model selection among 30 variants, when forecasting is primary goal, when you need proper state space framework for intervals

**In practice:** Both often give similar forecasts. Use whichever is easier to implement and interpret for your context.

### Q5: How do you choose between classical methods and machine learning?

**Answer:**

| Factor | Classical (ARIMA/ETS) | ML (RF, LSTM, etc.) |
|--------|----------------------|---------------------|
| Data size | Small-medium | Large |
| Interpretability | High | Low |
| Uncertainty | Well-calibrated intervals | Harder to quantify |
| Multiple series | Separate models | Can share patterns |
| Complex patterns | Limited | Can learn anything |
| Computational cost | Low | High |

**Rule of thumb:** Start with classical baselines. Use ML if you have lots of data AND classical methods underperform.

### Q6: What information criteria do you use for model selection? Explain AIC vs BIC.

**Answer:**
- **AIC = -2ln(L) + 2k:** Optimizes prediction error; can overfit with large samples
- **BIC = -2ln(L) + k·ln(n):** Consistent (selects true model); more parsimonious
- **AICc:** Corrected AIC for small samples (use when n/k < 40)

**Guideline:** Use AICc for forecasting, BIC for inference/interpretation.

## Practical Challenges

### Q7: How do you handle missing values in time series?

**Answer:**
1. **Linear interpolation:** For sporadic missing values
2. **Forward/backward fill:** When last known value is reasonable
3. **Seasonal imputation:** Fill with same period from previous cycle
4. **Model-based:** Kalman filter or EM algorithm
5. **Missing indicator:** Add binary feature, let model learn

**Never:** Delete rows (breaks temporal continuity) or use mean imputation globally.

### Q8: What is data leakage in time series? How do you prevent it?

**Answer:** Using future information when training or creating features.

**Common sources:**
- Random train-test split (future in training set)
- Rolling features without shift (includes current value)
- Scaling on full data (test statistics in training)
- External features not available at forecast time

**Prevention:**
- Always split temporally
- Use `.shift(1).rolling()` for features
- Fit scalers on training data only
- Verify feature availability in production

### Q9: Explain multi-step forecasting strategies.

**Answer:**
1. **Recursive:** Use 1-step model, iterate (feed predictions back)
2. **Direct:** Train separate model for each horizon
3. **MIMO:** Single model outputs all horizons

**Trade-offs:**
- Recursive: One model, but errors accumulate
- Direct: No error accumulation, but h models needed
- MIMO: Balance, but needs careful architecture

**For ARIMA:** Recursive is standard and optimal.
**For ML:** Direct often better (avoids compounding errors).

### Q10: How do you evaluate forecast accuracy?

**Answer:**

**Metrics:**
- MAE: Easy to interpret, robust to outliers
- RMSE: Penalizes large errors more
- MAPE: Percentage errors, but undefined at zero
- MASE: Scale-free, compares to naive forecast (MASE < 1 is good)

**Evaluation:**
- Use rolling origin cross-validation
- Match evaluation horizon to business need
- Always compare to baselines (naive, seasonal naive)
- Check prediction interval coverage

## Advanced Topics

### Q11: What is the Kalman filter and when would you use it?

**Answer:** Recursive algorithm for optimal state estimation in linear Gaussian state space models.

**Use cases:**
- Tracking unobserved components (level, trend)
- Online filtering (update as data arrives)
- Missing data handling (natural framework)
- Time-varying parameters

**Connection:** ETS, ARIMA, structural time series are all state space models; Kalman filter provides unified estimation.

### Q12: Explain Granger causality. What are its limitations?

**Answer:** X Granger-causes Y if past X improves prediction of Y beyond Y's own past.

**Limitations:**
- **Not true causation:** Correlation due to common causes gives spurious GC
- **Requires stationarity:** Standard tests need stationary data
- **Sensitive to lag selection:** Results change with different lags
- **Omitted variable bias:** Missing Z that causes both X and Y
- **Contemporaneous effects missed:** Only tests lagged relationships

### Q13: When should you use deep learning for time series?

**Answer:** Consider DL when:
- Large dataset (thousands+ observations)
- Multiple related series (can share representations)
- Complex patterns (nonlinear, interaction effects)
- Long-range dependencies (attention mechanisms help)

**Avoid DL when:**
- Small dataset (ARIMA usually wins)
- Interpretability required
- Simple patterns (Occam's razor)
- Computational constraints

**Popular architectures:** LSTM, TCN (temporal CNN), Transformers

### Q14: How do you detect and handle structural breaks?

**Answer:**

**Detection:**
- Visual inspection
- Chow test (tests for break at known point)
- CUSUM (cumulative sum of residuals)
- PELT algorithm (multiple change-points)

**Handling:**
- Regime-switching models (Markov switching)
- Structural break dummies in regression
- Train only on post-break data
- Time-varying parameters

### Q15: What is forecast reconciliation?

**Answer:** Ensuring forecasts at different aggregation levels are consistent.

**Example:** Product forecasts should sum to category forecast, which sums to total.

**Approaches:**
- **Top-down:** Forecast aggregate, distribute
- **Bottom-up:** Forecast individuals, sum
- **Optimal reconciliation:** Combine all levels optimally (MinT approach)

**Why it matters:** Inconsistent forecasts confuse planning (inventory, budgets).

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What are the three components of the Box-Jenkins methodology?</summary>

<div class="answer">
<strong>Answer:</strong> Identification, Estimation, Diagnostic Checking

1. **Identification:** Determine (p, d, q) using ACF/PACF, stationarity tests
2. **Estimation:** Fit model using MLE or conditional least squares
3. **Diagnostic Checking:** Verify residuals are white noise (Ljung-Box, ACF plots)

Iterate if diagnostics fail: return to identification.

<div class="pitfall">
<strong>Common pitfall:</strong> Skipping diagnostics. A model with good AIC can still have autocorrelated residuals, indicating misspecification.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Explain the bias-variance trade-off in time series forecasting.</summary>

<div class="answer">
<strong>Answer:</strong>
- **Bias:** Systematic error from wrong model assumptions (underfitting)
- **Variance:** Sensitivity to training data (overfitting)

**In time series context:**
- Simple model (AR(1)): May miss patterns (high bias) but stable (low variance)
- Complex model (ARIMA(5,1,5)): Captures patterns but unstable (high variance)

**Manifestation:**
- High bias: Poor training fit, similar performance on test
- High variance: Great training fit, poor test performance

**Regularization:** AIC/BIC penalize complexity; cross-validation estimates out-of-sample error.

<div class="pitfall">
<strong>Common pitfall:</strong> Only looking at training error. Always evaluate on holdout or via cross-validation.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> What is the forecast function for ARIMA(0,1,1)? How does it relate to exponential smoothing?</summary>

<div class="answer">
<strong>Answer:</strong>
ARIMA(0,1,1): $(1-L)y_t = (1+\theta L)\epsilon_t$

Forecast function:
$$\hat{y}_{T+h|T} = y_T + \theta\hat{\epsilon}_T = y_T + \theta(y_T - \hat{y}_{T|T-1})$$

This equals Simple Exponential Smoothing with $\alpha = 1/(1+\theta)$:
$$\hat{y}_{T+h|T} = \alpha y_T + (1-\alpha)\hat{y}_{T|T-1}$$

**Connection:** SES is optimal for local level model / ARIMA(0,1,1).

<div class="pitfall">
<strong>Common pitfall:</strong> Treating ARIMA and ETS as completely different. They're deeply connected; many ETS models have ARIMA equivalents.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> How would you test for unit root in the presence of seasonality?</summary>

<div class="answer">
<strong>Answer:</strong>
Standard ADF tests for non-seasonal unit root. For seasonal unit root:

1. **HEGY test:** Tests for unit roots at seasonal and zero frequencies
2. **Canova-Hansen:** Tests null of stationarity at seasonal frequencies
3. **OCSB test:** Specifically for seasonal unit roots

**Practical approach:**
1. First, test for seasonal unit root (OCSB/HEGY)
2. If present, apply seasonal differencing $(1-L^s)$
3. Then test seasonally differenced series for regular unit root (ADF)
4. Apply regular differencing if needed

<div class="pitfall">
<strong>Common pitfall:</strong> Using ADF directly on seasonal data. ADF may reject unit root due to seasonality, not because series is stationary.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You're asked to forecast daily sales for a retailer. Walk through your approach.</summary>

<div class="answer">
<strong>Answer:</strong>

**1. Understand the data:**
- Length of history, granularity
- Missing values, outliers
- External factors (holidays, promotions)

**2. Explore patterns:**
- Plot series, decomposition
- Check for trend, weekly/annual seasonality
- Examine ACF/PACF

**3. Baseline models:**
- Naive (yesterday's sales)
- Seasonal naive (same day last week)
- Moving average

**4. Candidate models:**
- SARIMA (captures ARMA + seasonality)
- ETS (automatic model selection)
- Prophet (handles holidays, easy to use)
- XGBoost with lag features (if many series)

**5. Evaluation:**
- Rolling origin CV (last 8-12 weeks)
- Metrics: MAE, MAPE, MASE
- Compare to baselines

**6. Production considerations:**
- Forecast horizon needed
- Update frequency
- Uncertainty communication
- Monitoring and retraining schedule

<div class="pitfall">
<strong>Common pitfall:</strong> Jumping to complex models. Start simple (seasonal naive is often hard to beat) and add complexity only if justified by evaluation.
</div>
</div>
</details>

## References

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley.
3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer.
