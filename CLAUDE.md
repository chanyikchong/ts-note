You are acting as a senior software engineer and technical writer. Build a complete, working project in the current repository that implements a bilingual (English/中文) Time Series study note system that is interview-ready, mathematically rigorous, easy to navigate, and continuously extensible. You must implement the full system (docs + code examples + site/app + Q&A/update workflow + skills extraction + quizzes).

============================================================
0) HARD RULES
============================================================
- Create and use a folder named ".claude/" for any temporary files, caches, intermediate artifacts, or scratch work your build process needs.
- Do NOT store final user-facing content exclusively in ".claude/". ".claude/" is for temporary/internal artifacts only.
- The repository must be runnable locally offline (except optional LLM integrations). Provide clear setup/run steps.
- Prefer open formats: Markdown for notes, executable Python for examples, JSON/YAML for metadata.
- Every major design decision must be recorded in a short "DECISIONS.md".

CLARIFICATION POLICY:
- If you encounter a blocking ambiguity that materially affects correctness (e.g., choice of doc framework, packaging tool, bilingual routing rules, code framework), you MUST ask the user a concise clarifying question before proceeding.
- If the ambiguity is non-blocking, choose a reasonable default, proceed, and record the assumption in DECISIONS.md.
- Ask at most 3 clarification questions total, each with 2–4 concrete options and a recommended default.

============================================================
1) PROJECT GOAL
============================================================
Create a bilingual Time Series knowledge base and learning system with:
1) Comprehensive study notes sufficient to study time series analysis/forecasting and pass interviews, including mathematical explanations.
2) Strong organization: table of contents, navigation, cross-links, search, and consistent structure.
3) Easy extensibility: user can keep adding new topics/notes/code.
4) Example code for core methods (clean, minimal, correct, runnable).
5) Language switch: English <-> Chinese with parallel structure.
6) Q&A on content + workflow to update content to resolve questions (local tooling; optional LLM plugin).
7) ".claude/" folder usage for temporary files (strict).
8) Extract a reusable “content generation procedure + requirement standard” as “skills” in a /skills folder.
9) A quiz section for each topic with answers hidden by default; user can click to reveal the answer, and each answer includes a clear explanation.

============================================================
2) USER-FACING OUTPUTS (WHAT TO BUILD)
============================================================
A) A documentation website (static) with:
- Table of Contents (global and per-section)
- Sidebar navigation by topic
- Full-text search (local)
- Language toggle (EN/中文)
- Clean reading experience for math (KaTeX or MathJax)
- Interactive “click to reveal answer” quizzes (no server required)
Implementation default: MkDocs Material or Docusaurus.
Choose one, justify in DECISIONS.md, then implement.

Quiz UI requirement (STRICT):
- Each topic page must include a “Quiz” section at the end.
- Each quiz must contain at least 5 questions:
  - 2 conceptual questions
  - 2 math/derivation questions
  - 1 practical/debugging/modeling question
- Answers must be hidden by default and revealed on click using a static-site-compatible mechanism:
  - Prefer Markdown-native collapsibles:
    - HTML <details><summary>Question</summary>Answer...</details>
    - Or framework-specific collapsible blocks (document choice)
- When revealed, the answer must include:
  - direct answer
  - explanation (why)
  - if math is involved, include the key equation(s)
  - a common pitfall/misconception

B) Time Series Notes (content) covering interview-level breadth + math:
- Must include at minimum:
  - Foundations: stationarity, ergodicity (high-level), autocovariance/autocorrelation, partial autocorrelation
  - Time-domain models: AR, MA, ARMA, ARIMA, SARIMA; identification (ACF/PACF), estimation (MLE/LS), diagnostics
  - Exponential smoothing / ETS: simple, Holt, Holt-Winters; state-space interpretation (high-level)
  - Decomposition: STL, trend/seasonality/residual; handling calendar effects
  - Forecasting: prediction intervals, multi-step forecasting strategies, rolling-origin evaluation
  - Model selection & validation: AIC/BIC, residual tests, cross-validation for time series, leakage pitfalls
  - Spectral analysis: periodogram intuition, frequency domain basics (no excessive depth, but correct)
  - State space models + Kalman filter (core equations + intuition)
  - Multivariate TS: VAR/VARMA basics, Granger causality (concept + caveats)
  - Change-point detection and anomaly detection (classical + simple approaches)
  - Time series features and classical pipelines (scaling, missing data, outliers)
  - Deep learning for TS (overview): RNN/LSTM/TCN, Transformer-style forecasting, when/why they help, failure modes
  - Practical modeling: data splitting, backtesting, uncertainty, deployment concerns
  - Common interview questions section with concise answers
- Each topic must have:
  1) Interview summary (3–6 lines)
  2) Core definitions
  3) Math and derivations
  4) Algorithm/model sketch
  5) Common pitfalls
  6) Mini example
  7) Quiz (>=5 hidden-answer questions)
  8) References
- Create mirrored bilingual structure:
  docs/en/... and docs/zh/...
  Keep the same file tree and section ordering for both languages.
- Quiz sections must be bilingual and aligned (same number of questions per mirrored page).

C) Example Code Library (Python):
- Provide runnable reference implementations and small demos (prefer numpy + statsmodels where useful, but keep offline-friendly):
  - Simulate AR/MA/ARMA/ARIMA processes
  - Fit ARIMA/SARIMA (if dependency allowed) + diagnostics
  - ETS/Holt-Winters forecasting
  - STL decomposition
  - Kalman filter for local level / local trend model
  - VAR example (multivariate)
  - Change-point detection (simple offline algorithm)
  - Rolling backtest + metrics (MAE, RMSE, MAPE/SMAPE, MASE)
- Include README per method with:
  - How it relates to notes
  - How to run
  - Expected behavior/output
- Provide a single entrypoint to run demos, e.g. `python -m ts_examples.run --demo arima`.

D) “Ask Questions + Update Content” Workflow (offline-first):
Implement a local workflow that lets the user:
- Ask a question referencing a note section
- Receive an answer via local retrieval + templated response
- Generate a patch proposal to update the relevant markdown pages
Minimum viable implementation (must be fully offline):
- A CLI tool `qa/ask.py` that:
  - indexes docs markdown locally (simple BM25)
  - retrieves relevant sections
  - produces a structured answer template:
    - direct answer
    - step-by-step reasoning summary
    - alternative perspectives
    - action plan
  - outputs a suggested content patch under `.claude/patches/` (temp) AND a final patch under `proposals/`
- The Q&A tool should optionally propose adding/adjusting quiz questions if it detects user confusion around a concept (rule-based heuristics are fine).

Optional LLM integration:
- Provide a plugin interface to call an LLM (disabled by default)
- Document how to enable with an env var, but keep offline path working.

E) Skills Extraction System:
- Create `skills/` folder.
- Create a “standard skill creation template” (Markdown) defining:
  - Skill name, scope, prerequisites
  - Inputs/outputs
  - Procedure
  - Quality checklist
  - Common failure modes
  - Examples
- Extract from this Time Series knowledge-base-building process reusable skills:
  - "Create a Bilingual, Interview-Ready Time Series Study Note Knowledge Base (with quizzes)"
  - "Time Series Notes with Math + Code + Backtesting + Quizzes"

============================================================
3) REPOSITORY STRUCTURE (REQUIRED)
============================================================
Create this structure (add files as needed):
- docs/
  - en/
  - zh/
  - assets/
- ts_examples/
  - demos/
  - run.py (or run/__init__.py)
- qa/
  - ask.py
  - index.py
  - patch.py
  - validate.py
- proposals/
- skills/
  - template.md
  - <skill_1>.md
  - <skill_2>.md
- .claude/ (temp only; but must exist)
- mkdocs.yml (or docusaurus config)
- README.md
- DECISIONS.md
- requirements.txt (or package manager config)
- Makefile (optional) or scripts/

============================================================
4) CONTENT STANDARDS (STRICT)
============================================================
For each note page (EN and ZH), use this fixed section order:
1) Interview summary (3–6 lines)
2) Core definitions
3) Math and derivations
4) Algorithm/model sketch
5) Common pitfalls
6) Mini example
7) Quiz (>= 5 questions; answers hidden; explained)
8) References

============================================================
5) NAVIGATION REQUIREMENTS
============================================================
- Global TOC must include all topics.
- Language toggle must preserve the current page context (EN page -> corresponding ZH page).
- Search must work across the current language content at minimum.

============================================================
6) TESTING / VALIDATION
============================================================
Add basic automated checks (must be runnable locally):
- link checker (internal links)
- markdown lint (or minimal custom validation)
- ensure EN and ZH trees match (same relative paths)
- ensure required sections exist per page
- ensure each topic page includes a "Quiz" section with >= 5 <details> (or equivalent)
- ensure EN and ZH quiz counts match per mirrored page

Provide a command to run all checks:
- `python -m qa.validate`

============================================================
7) DELIVERABLES
============================================================
You must produce:
- Working site build and serve instructions
- Working example code execution instructions
- Working Q&A + patch proposal instructions
- Skills in /skills
- Strict compliance with ".claude/" temporary usage rule
- A final summary in README.md:
  - what’s included
  - how to extend with new notes
  - how to add bilingual pages
  - how to add new demos and connect them
  - how to add quizzes and the required quiz format

============================================================
8) IMPLEMENTATION PLAN (FOLLOW THIS)
============================================================
Step 1: Initialize repository structure and configs.
Step 2: Build docs site scaffolding + language switch + quiz reveal UI.
Step 3: Write the Time Series note pages (EN first, then ZH mirrored), including quizzes.
Step 4: Implement demos and entrypoints.
Step 5: Implement QA indexing + retrieval + patch proposal pipeline (including quiz update suggestions).
Step 6: Add validators/tests (including quiz checks).
Step 7: Create skills template + extracted skills.
Step 8: Verify all acceptance criteria.

============================================================
9) ACCEPTANCE CRITERIA (MUST PASS)
============================================================
- `docs/en` and `docs/zh` have mirrored file trees for all core Time Series topics.
- Local site can be served and renders math correctly.
- Language toggle exists and maps to parallel page.
- Search works (at least per-language).
- Every topic page includes a Quiz section with >= 5 hidden-answer questions; answers reveal on click and include explanations.
- EN and ZH quiz sections match in count per mirrored page.
- At least 8 runnable demos exist (ARIMA, ETS, STL, Kalman, VAR, change-point, rolling backtest, metrics).
- `qa/ask.py` can retrieve relevant sections offline and generate a patch proposal in `proposals/`.
- `skills/template.md` exists and is used by at least two skills in `skills/`.
- ".claude/" exists and is used only for temporary artifacts; no final content must depend on it.
- README.md includes precise commands to run everything.

Now implement the project completely. Create files and write all necessary content/code/config. Ensure everything is internally consistent and runnable.
If and only if you hit a blocking ambiguity, ask up to 3 concise clarifying questions as specified in the CLARIFICATION POLICY, then continue.

