# Time Series Study Notes

A comprehensive bilingual (English/中文) study note system for time series analysis and forecasting, designed for interview preparation. Features mathematical rigor, interactive quizzes, and runnable code examples.

## Features

- **Bilingual Content**: Complete coverage in English and Chinese with synchronized structure
- **Interview-Ready**: Concise summaries, key formulas, and common interview questions
- **Interactive Quizzes**: 5+ questions per topic with click-to-reveal answers
- **Code Examples**: Runnable Python demos for all major algorithms
- **Mathematical Rigor**: Proper derivations with LaTeX rendering via MathJax
- **Q&A System**: Offline search and content improvement workflow

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Serve Documentation Locally

```bash
mkdocs serve
```

Visit http://localhost:8000 to view the site.

### 3. Run Code Examples

```bash
# List available demos
python -m ts_examples.run --help

# Run specific demo
python -m ts_examples.run --demo arima
python -m ts_examples.run --demo ets
python -m ts_examples.run --demo kalman
python -m ts_examples.run --demo var
python -m ts_examples.run --demo changepoint
python -m ts_examples.run --demo backtest
python -m ts_examples.run --demo metrics
python -m ts_examples.run --demo stl
```

### 4. Ask Questions (Q&A System)

```bash
# Ask a question about time series
python -m qa.ask "What is stationarity?"

# Get answer in specific language
python -m qa.ask "How does ARIMA work?" --language en

# Generate improvement patch proposal
python -m qa.ask "I don't understand the Kalman filter" --propose-patch
```

### 5. Validate Documentation

```bash
python -m qa.validate
```

## Project Structure

```
time_series/
├── docs/
│   ├── en/                    # English documentation
│   │   ├── foundations/       # Stationarity, autocorrelation
│   │   ├── time-domain/       # AR, MA, ARMA, ARIMA, SARIMA
│   │   ├── exponential-smoothing/  # SES, Holt, Holt-Winters, ETS
│   │   ├── decomposition/     # STL, classical decomposition
│   │   ├── forecasting/       # Prediction intervals, multi-step
│   │   ├── model-selection/   # AIC/BIC, cross-validation, diagnostics
│   │   ├── spectral/          # Spectral analysis
│   │   ├── state-space/       # Kalman filter
│   │   ├── multivariate/      # VAR, Granger causality
│   │   ├── change-detection/  # Change-point detection
│   │   ├── anomaly-detection/ # Anomaly detection
│   │   ├── features/          # Feature engineering
│   │   ├── deep-learning/     # RNN, LSTM, TCN, Transformers
│   │   ├── practical/         # Practical modeling tips
│   │   └── interview/         # Common interview questions
│   ├── zh/                    # Chinese documentation (mirrored)
│   ├── stylesheets/           # Custom CSS
│   └── javascripts/           # MathJax configuration
├── ts_examples/
│   ├── demos/                 # Runnable algorithm demos
│   │   ├── arima_demo.py
│   │   ├── ets_demo.py
│   │   ├── stl_demo.py
│   │   ├── kalman_demo.py
│   │   ├── var_demo.py
│   │   ├── changepoint_demo.py
│   │   ├── backtest_demo.py
│   │   └── metrics_demo.py
│   └── run.py                 # Unified demo runner
├── qa/
│   ├── ask.py                 # Q&A tool
│   ├── index.py               # BM25 document indexer
│   ├── patch.py               # Patch management
│   └── validate.py            # Documentation validator
├── proposals/                 # Generated patch proposals
├── skills/
│   ├── template.md            # Skill creation template
│   ├── bilingual-knowledge-base.md
│   └── bilingual-technical-notes.md
├── .claude/                   # Temporary artifacts (not for final content)
├── .github/
│   └── workflows/
│       └── deploy.yml         # GitHub Pages deployment
├── mkdocs.yml                 # MkDocs configuration
├── requirements.txt           # Python dependencies
├── DECISIONS.md               # Design decisions
└── README.md                  # This file
```

## Topics Covered

### Foundations
- Stationarity (strict, weak, tests)
- Autocorrelation and partial autocorrelation (ACF/PACF)

### Time Domain Models
- AR (autoregressive)
- MA (moving average)
- ARMA
- ARIMA
- SARIMA (seasonal)
- Model identification using ACF/PACF

### Exponential Smoothing
- Simple Exponential Smoothing
- Holt's method (trend)
- Holt-Winters (trend + seasonality)
- ETS framework (state space)

### Decomposition
- STL decomposition
- Classical decomposition

### Forecasting
- Prediction intervals
- Multi-step forecasting strategies

### Model Selection
- Information criteria (AIC, BIC, AICc)
- Time series cross-validation
- Residual diagnostics

### Advanced Topics
- Spectral analysis
- Kalman filter
- VAR models
- Granger causality
- Change-point detection
- Anomaly detection
- Deep learning for time series

## Adding New Content

### Adding a New Topic

1. Create English version in `docs/en/<category>/topic.md`
2. Create Chinese version in `docs/zh/<category>/topic.md`
3. Follow the required section order:
   - Interview Summary
   - Core Definitions
   - Math and Derivations
   - Algorithm/Model Sketch
   - Common Pitfalls
   - Mini Example
   - Quiz (5+ questions)
   - References
4. Add navigation entry in `mkdocs.yml`
5. Run `python -m qa.validate` to verify

### Adding a Quiz Question

Use this format:

```html
<details class="quiz">
<summary><strong>Q1 (Conceptual):</strong> Your question here?</summary>

<div class="answer">
<strong>Answer:</strong> Direct answer

<strong>Explanation:</strong> Why this is the answer

**Key equation:** $relevant\_equation$ (if applicable)

<div class="pitfall">
<strong>Common pitfall:</strong> Common misconception to avoid
</div>
</div>
</details>
```

### Adding a Code Demo

1. Create `ts_examples/demos/your_demo.py`
2. Add import in `ts_examples/demos/__init__.py`
3. Add case in `ts_examples/run.py`
4. Test with `python -m ts_examples.run --demo your_demo`

## Deployment to GitHub Pages

The project includes GitHub Actions workflow for automatic deployment:

1. Push to `main` or `master` branch
2. GitHub Actions builds and deploys to GitHub Pages
3. Access at: `https://<username>.github.io/time-series-notes/`

To deploy manually:

```bash
mkdocs gh-deploy
```

## Q&A Workflow

The Q&A system enables offline retrieval and content improvement:

```bash
# Build/rebuild index
python -m qa.index --rebuild

# Ask questions
python -m qa.ask "What is the difference between AR and MA?"

# With patch proposal
python -m qa.ask "I'm confused about stationarity" --propose-patch

# Manage patches
python -m qa.patch list
python -m qa.patch show <patch_id>
python -m qa.patch apply <patch_id>
```

## Validation

Run validation to ensure content quality:

```bash
# Full validation
python -m qa.validate

# Errors only
python -m qa.validate --errors-only

# JSON output
python -m qa.validate --json
```

Validation checks:
- EN/ZH file trees match
- Required sections present
- Quiz sections have 5+ questions
- EN/ZH quiz counts match
- Internal links valid
- Markdown syntax valid

## Tech Stack

- **Documentation**: MkDocs Material with i18n plugin
- **Math Rendering**: MathJax
- **Code Examples**: Python with NumPy, StatsModels, SciPy
- **Search**: BM25 (offline)
- **Deployment**: GitHub Pages via GitHub Actions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the content standards
4. Run validation: `python -m qa.validate`
5. Submit a pull request

## License

MIT License

## References

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*. Wiley.
- Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Springer.
