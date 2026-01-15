---
name: "Obsidian to GitHub Pages Publishing"
description: "Create and publish study notes from an Obsidian vault to a GitHub Pages website. Use when: (1) you have an Obsidian vault with markdown notes, (2) you want to publish notes as a static website, (3) you need a CI/CD workflow for automatic deployment, or (4) you want MkDocs Material with search, navigation, and math rendering."
---

# Obsidian to GitHub Pages Publishing

## Overview

Transform an Obsidian vault into a polished, searchable documentation website hosted on GitHub Pages. This skill covers the complete workflow from local Obsidian notes to a live website with automatic deployment.

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| Obsidian vault path | Yes | Path to your existing Obsidian vault |
| Repository name | Yes | GitHub repository name for hosting |
| Site name | No | Website title (default: repository name) |
| Theme | No | MkDocs theme (default: Material) |
| Enable math | No | Whether to enable LaTeX math (default: Yes) |
| Custom domain | No | Optional custom domain (e.g., notes.example.com) |

## Outputs

| Output | Format | Description |
|--------|--------|-------------|
| docs/ folder | Markdown | Cleaned/organized notes ready for MkDocs |
| mkdocs.yml | YAML | Site configuration |
| .github/workflows/deploy.yml | YAML | GitHub Actions workflow |
| requirements.txt | Text | Python dependencies |
| Live website | URL | https://username.github.io/repo-name/ |

## Procedure

### Step 1: Prepare Obsidian Vault Structure

Obsidian and MkDocs have different link/embed conventions. Prepare your vault:

```
obsidian-vault/
├── Topic1/
│   ├── subtopic1.md
│   └── subtopic2.md
├── Topic2/
│   └── content.md
├── _templates/        # Exclude from publishing
├── _attachments/      # Rename to assets/images/
└── index.md           # Home page
```

**Key conversions needed:**
- `[[wiki-links]]` → `[regular links](path.md)`
- `![[embedded-notes]]` → Include content or convert to links
- `![[image.png]]` → `![](assets/images/image.png)`
- Remove Obsidian-specific YAML frontmatter if incompatible

### Step 2: Create Project Structure

```bash
mkdir my-notes-site
cd my-notes-site

# Copy and organize content
cp -r /path/to/obsidian-vault/. docs/

# Create required files
touch mkdocs.yml
touch requirements.txt
mkdir -p .github/workflows
mkdir -p docs/assets/images
mkdir -p docs/assets/javascripts
```

### Step 3: Configure MkDocs

Create `mkdocs.yml`:

```yaml
site_name: My Study Notes
site_description: Personal study notes and documentation
site_url: https://username.github.io/repo-name/

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.details
  - admonition
  - tables
  - toc:
      permalink: true

extra_javascript:
  - assets/javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js

nav:
  - Home: index.md
  - Topic 1:
    - Subtopic 1: topic1/subtopic1.md
    - Subtopic 2: topic1/subtopic2.md
  - Topic 2: topic2/content.md
```

### Step 4: Enable MathJax (for LaTeX)

Create `docs/assets/javascripts/mathjax.js`:

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
```

### Step 5: Create GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build site
        run: mkdocs build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
        timeout-minutes: 20
```

### Step 6: Create Requirements File

Create `requirements.txt`:

```
mkdocs>=1.5.0
mkdocs-material>=9.0.0
pymdown-extensions>=10.0
```

### Step 7: Convert Obsidian-Specific Syntax

Use a script to convert wiki-links to standard markdown:

```python
#!/usr/bin/env python3
"""Convert Obsidian wiki-links to standard markdown links."""
import re
import os
from pathlib import Path

def convert_wiki_links(content, current_file):
    """Convert [[link]] to [link](link.md)"""
    # Pattern for wiki-links: [[note]] or [[note|alias]]
    pattern = r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]'

    def replacer(match):
        target = match.group(1)
        alias = match.group(2) or target
        # Convert spaces to dashes, make lowercase for URL
        target_path = target.lower().replace(' ', '-') + '.md'
        return f'[{alias}]({target_path})'

    return re.sub(pattern, replacer, content)

def convert_image_embeds(content):
    """Convert ![[image.png]] to ![](assets/images/image.png)"""
    pattern = r'!\[\[([^\]]+\.(png|jpg|jpeg|gif|svg))\]\]'

    def replacer(match):
        image = match.group(1)
        return f'![](assets/images/{image})'

    return re.sub(pattern, replacer, content, flags=re.IGNORECASE)

def process_file(filepath):
    """Process a single markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = convert_wiki_links(content, filepath)
    content = convert_image_embeds(content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    docs_path = Path('docs')
    for md_file in docs_path.rglob('*.md'):
        print(f'Processing: {md_file}')
        process_file(md_file)

if __name__ == '__main__':
    main()
```

### Step 8: Initialize Git and Push

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: study notes site"

# Create GitHub repository and push
gh repo create repo-name --public --source=. --push

# Or if repo exists:
git remote add origin git@github.com:username/repo-name.git
git push -u origin main
```

### Step 9: Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: GitHub Actions
3. The workflow will automatically build and deploy on push

### Step 10: Verify Deployment

After the workflow completes:
1. Check Actions tab for build status
2. Visit https://username.github.io/repo-name/
3. Test search, navigation, and math rendering

## Quality Checklist

- [ ] All wiki-links converted to standard markdown
- [ ] Images moved to docs/assets/images/ and paths updated
- [ ] mkdocs.yml has correct nav structure
- [ ] MathJax renders equations correctly
- [ ] Search works for content
- [ ] GitHub Actions workflow succeeds
- [ ] Site is accessible at GitHub Pages URL
- [ ] No broken internal links
- [ ] Code blocks have syntax highlighting
- [ ] Mobile responsive layout works

## Common Failure Modes

| Failure | Cause | Prevention |
|---------|-------|------------|
| Broken links after publishing | Wiki-links not converted | Run conversion script before building |
| Images not showing | Wrong path or not in docs/ | Move to docs/assets/images/, use relative paths |
| Math not rendering | MathJax not configured | Add mathjax.js and extra_javascript in mkdocs.yml |
| Build fails in Actions | Missing dependencies | Ensure requirements.txt has all packages |
| 404 on GitHub Pages | Pages not enabled | Enable Pages in Settings → Source: GitHub Actions |
| Deployment timeout | Large site | Increase timeout in workflow (timeout-minutes: 20) |
| Private content published | Didn't exclude folders | Add patterns to .gitignore or exclude in nav |

## Examples

### Example 1: Simple Note Conversion

**Input (Obsidian):**
```markdown
# Machine Learning Notes

See [[neural-networks]] for deep learning basics.

Related: [[backpropagation|Backprop Algorithm]]

![[gradient-descent.png]]

## Key Equations

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$
```

**Output (MkDocs-compatible):**
```markdown
# Machine Learning Notes

See [neural-networks](neural-networks.md) for deep learning basics.

Related: [Backprop Algorithm](backpropagation.md)

![](assets/images/gradient-descent.png)

## Key Equations

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$
```

### Example 2: Navigation Structure

**Obsidian folder structure:**
```
vault/
├── ML/
│   ├── supervised.md
│   └── unsupervised.md
├── Stats/
│   ├── probability.md
│   └── inference.md
└── index.md
```

**mkdocs.yml nav:**
```yaml
nav:
  - Home: index.md
  - Machine Learning:
    - Supervised Learning: ML/supervised.md
    - Unsupervised Learning: ML/unsupervised.md
  - Statistics:
    - Probability: Stats/probability.md
    - Statistical Inference: Stats/inference.md
```

## Obsidian-MkDocs Compatibility Reference

| Feature | Obsidian | MkDocs Material |
|---------|----------|-----------------|
| Wiki links | `[[page]]` | `[page](page.md)` |
| Link alias | `[[page\|text]]` | `[text](page.md)` |
| Embeds | `![[page]]` | Include manually or link |
| Images | `![[img.png]]` | `![](path/img.png)` |
| Callouts | `> [!note]` | `!!! note` |
| Math inline | `$x^2$` | `$x^2$` (same) |
| Math block | `$$x^2$$` | `$$x^2$$` (same) |
| Code blocks | ` ```python ` | ` ```python ` (same) |
| Tags | `#tag` | Remove or use YAML frontmatter |

## References

- MkDocs Material: https://squidfunk.github.io/mkdocs-material/
- GitHub Pages documentation: https://docs.github.com/en/pages
- GitHub Actions for MkDocs: https://github.com/actions/deploy-pages
- Obsidian Help: https://help.obsidian.md/
- obsidian-export (alternative converter): https://github.com/zoni/obsidian-export
