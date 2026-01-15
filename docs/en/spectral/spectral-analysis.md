# Spectral Analysis

<div class="interview-summary">
<strong>Interview Summary:</strong> Spectral analysis decomposes time series into frequency components. The periodogram estimates power at each frequency. Peaks indicate dominant cycles. For stationary series, spectral density is the Fourier transform of autocovariance. Key insight: AR produces smooth spectrum; MA produces peaks. Useful for detecting hidden periodicities and understanding cyclical behavior.
</div>

## Core Definitions

**Spectral Density:** For a stationary process, the spectral density $f(\omega)$ represents power at frequency $\omega$:
$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$

**Periodogram:** Sample estimate of spectral density:
$$I(\omega_j) = \frac{1}{n}\left|\sum_{t=1}^{n}y_t e^{-i\omega_j t}\right|^2$$

at Fourier frequencies $\omega_j = 2\pi j/n$ for $j = 0, 1, \ldots, n/2$.

**Key Frequencies:**
- $\omega = 0$: Mean level (zero frequency)
- $\omega = 2\pi/m$: Period of m time units
- $\omega = \pi$: Nyquist frequency (fastest observable cycle = 2 time units)

## Math and Derivations

### Fourier Transform Relationship

The autocovariance and spectral density are Fourier transform pairs:
$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$
$$\gamma(h) = \int_{-\pi}^{\pi}f(\omega)e^{i\omega h}d\omega$$

**Parseval's relation:**
$$\gamma(0) = \text{Var}(y_t) = \int_{-\pi}^{\pi}f(\omega)d\omega$$

Total variance decomposes across frequencies.

### Spectral Density of AR(1)

For $y_t = \phi y_{t-1} + \epsilon_t$:
$$f(\omega) = \frac{\sigma^2}{2\pi|1-\phi e^{-i\omega}|^2} = \frac{\sigma^2}{2\pi(1+\phi^2-2\phi\cos\omega)}$$

Properties:
- $\phi > 0$: Peak at $\omega = 0$ (low-frequency dominance)
- $\phi < 0$: Peak at $\omega = \pi$ (high-frequency dominance)

### Spectral Density of MA(1)

For $y_t = \epsilon_t + \theta\epsilon_{t-1}$:
$$f(\omega) = \frac{\sigma^2}{2\pi}|1+\theta e^{-i\omega}|^2 = \frac{\sigma^2}{2\pi}(1+\theta^2+2\theta\cos\omega)$$

### Period Detection

If periodogram has peak at $\omega_j$, the dominant period is:
$$T = \frac{2\pi}{\omega_j} = \frac{n}{j}$$

For monthly data with annual cycle: peak at $\omega = 2\pi/12 \approx 0.524$.

## Algorithm/Model Sketch

**Spectral Analysis Procedure:**

```
1. Remove mean (and trend if necessary)
   y_centered = y - mean(y)

2. Apply window (optional, reduces leakage)
   Common: Hanning, Hamming, Blackman

3. Compute FFT
   Y = FFT(y_centered)

4. Compute periodogram
   I[j] = |Y[j]|² / n

5. Smooth periodogram (optional)
   - Daniell kernel
   - Log-smoothing
   - Welch's method

6. Identify peaks
   - Compare to red/white noise baseline
   - Test significance (F-test against continuum)

7. Interpret
   - Peaks → dominant periods
   - Smooth decay → AR-like behavior
   - Flat → white noise
```

**Frequency to Period Conversion:**
$$\text{Period} = \frac{n}{\text{index}} = \frac{2\pi}{\omega}$$

## Common Pitfalls

1. **Spectral leakage**: Sharp peaks in true spectrum appear spread out due to finite sample. Use windowing to reduce.

2. **Confusing periodogram with spectral density**: Periodogram is inconsistent (doesn't converge). Smooth it for density estimation.

3. **Ignoring aliasing**: Cycles faster than Nyquist (period < 2) appear at wrong frequencies. Ensure adequate sampling.

4. **Non-stationarity**: Spectral analysis assumes stationarity. Trend causes low-frequency blow-up. Detrend first.

5. **Over-interpreting peaks**: Random fluctuations create spurious peaks. Test significance against noise baseline.

6. **Wrong frequency interpretation**: $\omega = 0.5$ doesn't mean period = 0.5. Period = $2\pi/0.5 \approx 12.6$.

## Mini Example

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate signal with known frequencies
np.random.seed(42)
n = 500
t = np.arange(n)

# Components: trend + period 50 + period 12 + noise
y = (0.01 * t +                          # trend
     5 * np.sin(2 * np.pi * t / 50) +    # period 50
     3 * np.sin(2 * np.pi * t / 12) +    # period 12
     np.random.randn(n))                  # noise

# Detrend
y_detrended = signal.detrend(y)

# Compute periodogram
freqs, psd = signal.periodogram(y_detrended, fs=1.0)

# Find peaks
peaks, _ = signal.find_peaks(psd, height=np.percentile(psd, 90))

print("Detected periods:")
for p in peaks:
    if freqs[p] > 0:
        period = 1 / freqs[p]
        print(f"  Frequency {freqs[p]:.4f} → Period {period:.1f}")

# Expected: peaks near period 50 and period 12
```

## Quiz

<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> What is the relationship between the autocovariance function and spectral density?</summary>

<div class="answer">
<strong>Answer:</strong> They are Fourier transform pairs.

$$f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^{\infty}\gamma(h)e^{-i\omega h}$$

$$\gamma(h) = \int_{-\pi}^{\pi}f(\omega)e^{i\omega h}d\omega$$

**Interpretation:**
- ACF describes correlation in time domain
- Spectral density describes power in frequency domain
- Same information, different representation

**Key insight:** At $h=0$:
$$\gamma(0) = \text{Var}(y_t) = \int f(\omega)d\omega$$

Total variance = integral of spectral density (power across all frequencies).

<div class="pitfall">
<strong>Common pitfall:</strong> Thinking frequency and time domain analyses give different information. They're equivalent representations—choose based on what's easier to interpret for your problem.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q2 (Conceptual):</strong> Why does an AR(1) with positive φ have peak at frequency zero?</summary>

<div class="answer">
<strong>Answer:</strong> Positive AR(1) coefficient creates persistence—values tend to stay above or below mean for extended periods. This translates to slow oscillations (low frequency).

**Mathematical explanation:**
Spectral density: $f(\omega) \propto \frac{1}{1+\phi^2-2\phi\cos\omega}$

At $\omega = 0$: $f(0) \propto \frac{1}{(1-\phi)^2}$ (maximum for $\phi > 0$)
At $\omega = \pi$: $f(\pi) \propto \frac{1}{(1+\phi)^2}$ (minimum for $\phi > 0$)

**Intuition:**
- $\phi > 0$: Today's value predicts tomorrow's → smooth, low-frequency behavior
- $\phi < 0$: Today's value predicts opposite tomorrow → choppy, high-frequency behavior

<div class="pitfall">
<strong>Common pitfall:</strong> Expecting all AR processes to look similar in spectrum. The sign and magnitude of φ drastically change spectral shape.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q3 (Math):</strong> Derive the Nyquist frequency and explain why frequencies above it cannot be detected.</summary>

<div class="answer">
<strong>Answer:</strong> Nyquist frequency = $\pi$ rad/sample = 0.5 cycles/sample.

**Derivation:**
To observe a cycle, we need at least 2 samples per period (sample at peak and trough).

Minimum period detectable = 2 sample intervals
Maximum frequency = 1/(2 sample intervals) = 0.5 cycles/sample

In angular frequency: $\omega_{Nyquist} = 2\pi \times 0.5 = \pi$

**Aliasing:**
A signal with frequency $\omega > \pi$ appears as frequency $2\pi - \omega$ (reflected).

Example: True frequency 0.6 cycles/sample appears as 0.4 cycles/sample.

**Consequence:** Without higher sampling rate, we cannot distinguish $\omega$ from $2\pi - \omega$.

<div class="pitfall">
<strong>Common pitfall:</strong> Trying to detect daily cycles from monthly data. Monthly sampling (Nyquist = 2 months period) cannot see anything faster than bimonthly oscillations.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q4 (Math):</strong> Why is the raw periodogram an inconsistent estimator of spectral density?</summary>

<div class="answer">
<strong>Answer:</strong> The periodogram variance doesn't decrease as sample size increases.

**Technical explanation:**
$$\text{Var}(I(\omega)) \approx f(\omega)^2$$

for $\omega \neq 0, \pi$. The variance equals the squared mean—relative error stays constant!

**Why this happens:**
- Periodogram at each frequency uses information from the entire series
- But at each Fourier frequency, we essentially have one "observation"
- More data → more frequencies, but still one estimate per frequency

**Solution:** Smooth the periodogram
- Average nearby frequencies (reduces variance)
- Or use multitaper methods
- Trade bias for variance

<div class="pitfall">
<strong>Common pitfall:</strong> Treating raw periodogram peaks as definitive. Peaks can be noise; always smooth or test significance.
</div>
</div>
</details>

<details class="quiz">
<summary><strong>Q5 (Practical):</strong> You analyze hourly temperature data and see a strong peak at period 24 hours, but also unexpected peaks at periods 12, 8, 6 hours. What's happening?</summary>

<div class="answer">
<strong>Answer:</strong> These are **harmonics** of the fundamental daily cycle.

**Explanation:**
A pure 24-hour cycle would show only one peak at period 24. But real temperature patterns aren't pure sinusoids—they have:
- Sharp morning rise
- Gradual afternoon decline
- These non-sinusoidal shapes require multiple frequencies to represent

**Fourier's theorem:** Any periodic signal is a sum of harmonics:
$$y(t) = \sum_{k=1}^{\infty} a_k \cos(2\pi kt/24) + b_k \sin(2\pi kt/24)$$

Harmonics at periods 24/2=12, 24/3=8, 24/4=6, etc.

**Interpretation:**
- Period 24: Fundamental daily cycle
- Period 12: Asymmetry (morning ≠ evening)
- Period 8, 6: Further shape details

**Action:** This is normal for non-sinusoidal cycles. Focus on fundamental; harmonics indicate shape.

<div class="pitfall">
<strong>Common pitfall:</strong> Interpreting harmonics as separate physical phenomena. They're mathematical artifacts of non-sinusoidal shape.
</div>
</div>
</details>

## References

1. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer. Chapter 4.
2. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer. Chapter 4.
3. Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.
4. Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical Applications*. Cambridge University Press.
