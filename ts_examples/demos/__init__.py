"""
Time Series Demo Modules
========================

Each demo module provides a run_demo() function that demonstrates
a specific time series concept with:
- Data generation
- Model fitting/analysis
- Visualization (saved to .claude/)
- Educational output

Available demos:
- arima_demo: ARIMA modeling
- ets_demo: Exponential smoothing
- stl_demo: STL decomposition
- kalman_demo: Kalman filtering
- var_demo: Vector autoregression
- changepoint_demo: Change-point detection
- backtest_demo: Rolling backtesting
- metrics_demo: Forecast metrics
"""

from . import arima_demo
from . import ets_demo
from . import stl_demo
from . import kalman_demo
from . import var_demo
from . import changepoint_demo
from . import backtest_demo
from . import metrics_demo

DEMOS = {
    'arima': arima_demo,
    'ets': ets_demo,
    'stl': stl_demo,
    'kalman': kalman_demo,
    'var': var_demo,
    'changepoint': changepoint_demo,
    'backtest': backtest_demo,
    'metrics': metrics_demo,
}

__all__ = ['DEMOS', 'arima_demo', 'ets_demo', 'stl_demo', 'kalman_demo',
           'var_demo', 'changepoint_demo', 'backtest_demo', 'metrics_demo']
