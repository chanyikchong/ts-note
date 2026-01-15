# Time Series Study Notes

<div class="interview-summary">
<strong>Welcome to the Time Series Study Notes</strong> — a comprehensive, bilingual (English/中文) resource for learning time series analysis and forecasting, designed for interview preparation and practical application.
</div>

## What's Included

This knowledge base covers:

- **Foundations**: Stationarity, autocorrelation, partial autocorrelation
- **Time Domain Models**: AR, MA, ARMA, ARIMA, SARIMA with identification and estimation
- **Exponential Smoothing**: SES, Holt, Holt-Winters, ETS framework
- **Decomposition**: STL, classical decomposition, handling seasonality
- **Forecasting**: Prediction intervals, multi-step strategies, rolling evaluation
- **Model Selection**: AIC/BIC, cross-validation for time series, residual diagnostics
- **Spectral Analysis**: Periodogram, frequency domain basics
- **State Space Models**: Kalman filter, local level and trend models
- **Multivariate TS**: VAR, VARMA, Granger causality
- **Change Detection**: Change-point and anomaly detection methods
- **Feature Engineering**: Classical pipelines, scaling, missing data handling
- **Deep Learning**: RNN/LSTM/TCN, Transformers for time series
- **Practical Modeling**: Backtesting, deployment, common pitfalls

## Page Structure

Every topic page follows a consistent 8-section format:

1. **Interview Summary** — Key points in 3-6 lines
2. **Core Definitions** — Essential terminology and concepts
3. **Math and Derivations** — Rigorous mathematical foundations
4. **Algorithm/Model Sketch** — How the method works
5. **Common Pitfalls** — Mistakes to avoid
6. **Mini Example** — Quick illustration
7. **Quiz** — 5+ questions with hidden answers (click to reveal)
8. **References** — Further reading

## Getting Started

Choose a topic from the sidebar to begin. Each page is self-contained but builds on foundational concepts.

**Recommended learning path for beginners:**

1. Start with [Stationarity](foundations/stationarity.md) and [Autocorrelation](foundations/autocorrelation.md)
2. Move to [AR Models](time-domain/ar.md) → [MA Models](time-domain/ma.md) → [ARMA](time-domain/arma.md) → [ARIMA](time-domain/arima.md)
3. Learn [Model Identification](time-domain/identification.md) and [Residual Diagnostics](model-selection/residual-diagnostics.md)
4. Explore [Exponential Smoothing](exponential-smoothing/ses.md) and [Decomposition](decomposition/stl.md)
5. Advance to [State Space Models](state-space/kalman-filter.md) and [Multivariate TS](multivariate/var.md)

## Code Examples

Runnable Python demos are available in the `ts_examples/` directory. Run them with:

```bash
python -m ts_examples.run --demo <demo_name>
```

Available demos: `arima`, `ets`, `stl`, `kalman`, `var`, `changepoint`, `backtest`, `metrics`

## Language Toggle

Use the language selector in the header to switch between English and 中文. The site maintains parallel content in both languages.

---

*This is an open, extensible knowledge base. See the repository README for instructions on adding new content.*
