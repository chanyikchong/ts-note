# Design Decisions

This document records major design decisions made during project implementation.

## 1. Documentation Framework: MkDocs Material

**Decision**: Use MkDocs with Material theme instead of Docusaurus.

**Rationale**:
- Native Python ecosystem (matches our demo code)
- Excellent i18n plugin for bilingual support with automatic language switching
- Built-in search that works per-language
- First-class KaTeX/MathJax support via pymdownx.arithmatex
- Native `<details>` support + admonitions for quiz collapsibles
- Simpler configuration (YAML vs JavaScript)
- Lower dependency footprint (no Node.js required)

**Trade-offs**:
- Docusaurus has richer React-based interactivity (not needed here)
- MkDocs is simpler but sufficient for our static content needs

## 2. Math Rendering: KaTeX via pymdownx.arithmatex

**Decision**: Use KaTeX over MathJax.

**Rationale**:
- Faster rendering (important for math-heavy pages)
- Smaller bundle size
- Sufficient LaTeX coverage for time series formulas

## 3. Quiz Implementation: HTML `<details>` elements

**Decision**: Use native HTML `<details><summary>` for quiz answer reveal.

**Rationale**:
- No JavaScript required (works offline)
- Markdown-native, supported by MkDocs Material
- Accessible and semantic HTML
- Works in all modern browsers

## 4. Q&A Retrieval: BM25 (rank_bm25)

**Decision**: Use BM25 algorithm via `rank_bm25` library.

**Rationale**:
- Fully offline, no external API
- Simple and effective for document retrieval
- Pure Python implementation
- No large model downloads required

## 5. Python Version: 3.9+

**Decision**: Target Python 3.9 and above.

**Rationale**:
- Wide compatibility
- Type hints support
- All required libraries available

## 6. Package Management: pip + requirements.txt

**Decision**: Use simple requirements.txt over poetry/pipenv.

**Rationale**:
- Maximum portability
- No additional tooling required
- Clear and simple for users

## 7. Content Structure: Fixed 8-Section Template

**Decision**: Every topic page follows the same 8-section structure.

**Rationale**:
- Consistent learning experience
- Easy to validate programmatically
- Mirrors interview preparation needs
- Makes bilingual alignment straightforward

## 8. Bilingual Strategy: Parallel File Trees

**Decision**: Maintain docs/en/ and docs/zh/ with identical file structures.

**Rationale**:
- MkDocs i18n plugin expects this structure
- Easy to verify alignment
- Clear mapping between languages
- Supports language toggle preserving page context
