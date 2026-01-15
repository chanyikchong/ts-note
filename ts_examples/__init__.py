"""
Time Series Examples Library
============================

A collection of runnable demos for time series analysis concepts.

Available demos:
- arima: ARIMA model fitting, diagnostics, and forecasting
- ets: Exponential smoothing (SES, Holt, Holt-Winters)
- stl: STL decomposition
- kalman: Kalman filter for local level/trend models
- var: Vector Autoregression and Granger causality
- changepoint: Change-point detection algorithms
- backtest: Rolling-origin backtesting
- metrics: Forecast accuracy metrics

Usage:
    python -m ts_examples.run --demo arima
    python -m ts_examples.run --list
    python -m ts_examples.run --all
"""

__version__ = "1.0.0"
__author__ = "Time Series Notes Project"
