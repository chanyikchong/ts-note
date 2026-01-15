---
name: "Bilingual Interview-Ready Study Note Knowledge Base"
description: "Create a comprehensive bilingual (English/Chinese) study note knowledge base with interactive quizzes, optimized for technical interview preparation. Use when building documentation sites for learning technical subjects (ML, systems, algorithms) with: (1) bilingual content requirement, (2) interview preparation focus, (3) mathematical rigor needs, (4) interactive quiz requirements, or (5) code example integration."
---

# Bilingual Interview-Ready Study Note Knowledge Base

## Overview

Build a complete bilingual documentation system for technical study notes that includes mathematical content, code examples, interactive quizzes, and interview-focused summaries. Outputs a static site with language switching, search, and structured navigation.

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| Subject domain | Yes | Technical subject area (e.g., "Reinforcement Learning", "Distributed Systems") |
| Topic list | Yes | List of topics to cover with rough hierarchy |
| Primary language | No | EN or ZH (default: EN first, then mirror to ZH) |
| Code framework | No | PyTorch, TensorFlow, or language-agnostic (default: PyTorch) |
| Doc framework | No | MkDocs Material or Docusaurus (default: MkDocs Material) |

## Outputs

| Output | Format | Description |
|--------|--------|-------------|
| docs/en/ | Markdown | English documentation with all topics |
| docs/zh/ | Markdown | Chinese documentation mirroring EN structure |
| Code examples | Python | Runnable algorithm implementations |
| mkdocs.yml | YAML | Site configuration with i18n |
| Quiz sections | HTML | Click-to-reveal Q&A in each topic |

## Procedure

### Step 1: Initialize Project Structure

Create directory structure:
```
project/
├── docs/
│   ├── en/           # English content
│   ├── zh/           # Chinese content (mirrored)
│   └── assets/       # Shared assets (images, JS)
├── examples/         # Code examples
├── mkdocs.yml        # Site configuration
├── requirements.txt  # Dependencies
└── README.md
```

Configure MkDocs with:
- Material theme
- i18n plugin for language switching
- MathJax/KaTeX for equations
- Search plugin
- Code highlighting

### Step 2: Define Content Structure per Topic

Each topic page MUST follow this section order:

1. **Interview Summary** (3-6 lines)
   - Key concepts an interviewer expects you to know
   - "What to memorize" callout

2. **Core Definitions**
   - Formal definitions with notation
   - Tables for related concepts

3. **Math and Derivations**
   - Key equations with LaTeX
   - Step-by-step derivation where needed
   - Explanation of each equation's meaning

4. **Algorithm Sketch**
   - Pseudocode or algorithm box
   - Complexity analysis

5. **Common Pitfalls**
   - Numbered list of mistakes
   - How to avoid each

6. **Mini Example**
   - Concrete worked example
   - Small enough to trace by hand

7. **Quiz** (minimum 5 questions)
   - 2 conceptual questions
   - 2 math/derivation questions
   - 1 practical/debugging question
   - Use `<details><summary>` for click-to-reveal

8. **References**
   - Books, papers, courses
   - "What to memorize for interviews"

### Step 3: Write English Content First

For each topic:
1. Start with Interview Summary — forces clarity on essentials
2. Write Core Definitions with precise notation
3. Add Math with explanations per equation
4. Create Algorithm Sketch with pseudocode
5. List Common Pitfalls from experience
6. Develop Mini Example that's traceable
7. Write 5+ Quiz questions with detailed answers
8. Add References and memorization notes

### Step 4: Create Chinese Translation

Mirror structure exactly:
- Same file paths: `en/topic.md` → `zh/topic.md`
- Same section order and headings (translated)
- Same number of quiz questions
- Keep equations in LaTeX (universal)
- Keep code examples in English

Translation guidelines:
- Technical terms: use standard Chinese translations
- Keep English acronyms (MDP, TD, DQN)
- Mathematical notation unchanged
- Code comments can remain in English

### Step 5: Implement Code Examples

For each algorithm:
1. Create minimal, runnable implementation
2. Include docstring linking to docs
3. Add README with:
   - How it relates to notes
   - How to run
   - Expected output
4. Create unified entry point: `python -m examples.run --algo [name]`

### Step 6: Add Navigation and Cross-linking

- Global TOC in mkdocs.yml
- Per-section navigation
- Algorithm pages link to code
- Code READMEs link back to docs
- Language toggle preserves page context

### Step 7: Validate

Run validation checks:
- [ ] EN and ZH file trees match
- [ ] All required sections present
- [ ] Each topic has 5+ quiz questions
- [ ] EN/ZH quiz counts match
- [ ] Internal links valid
- [ ] Math renders correctly
- [ ] Code examples run

## Quality Checklist

- [ ] Every topic has all 8 required sections
- [ ] Interview Summary is 3-6 lines, actionable
- [ ] Math equations have explanations
- [ ] Quiz answers include: direct answer, explanation, key equation, common pitfall
- [ ] EN and ZH are structurally identical
- [ ] Code examples are minimal but complete
- [ ] Cross-links work between docs and code
- [ ] Site builds and serves locally
- [ ] Search works per-language
- [ ] Language toggle works

## Common Failure Modes

| Failure | Cause | Prevention |
|---------|-------|------------|
| Inconsistent EN/ZH structure | Adding content to one without other | Always edit both simultaneously |
| Quiz too easy | Surface-level questions | Include math derivation and practical debugging Qs |
| Math without explanation | Copy-pasting equations | Add "meaning" sentence after each equation |
| Code not runnable | Missing dependencies | Test in fresh environment |
| Broken cross-links | Path changes | Run link validator before publish |
| Interview Summary too long | Including everything | Force 3-6 line limit, focus on "must know" |

## Examples

### Example 1: Q-Learning Topic Page Structure

**Input:** Topic "Q-Learning" for RL knowledge base

**Output structure:**
```markdown
# Q-Learning

## Interview Summary
Q-learning is off-policy TD control that learns Q* directly...
**What to memorize**: Update rule, off-policy nature, convergence conditions.

## Core Definitions
### Q-Learning Update Rule
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]$$

## Math and Derivations
[Derivation from Bellman optimality...]

## Algorithm Sketch
```
For each episode:
  While not terminal:
    A = ε-greedy(Q, S)
    S', R = env.step(A)
    Q(S,A) ← Q(S,A) + α[R + γ max Q(S',·) - Q(S,A)]
    S ← S'
```

## Common Pitfalls
1. **Overestimation bias**: max over noisy estimates...

## Mini Example
[2-state MDP walkthrough...]

## Quiz
<details>
<summary><strong>Q1:</strong> What makes Q-learning "off-policy"?</summary>
**Answer**: Uses max instead of sampled action...
</details>
[4 more questions...]

## References
- Watkins & Dayan (1992)
- Sutton & Barto Ch. 6.5
```

### Example 2: Quiz Question Format

**Input:** Need quiz question about Bellman equation

**Output:**
```html
<details>
<summary><strong>Q3 (Math):</strong> Derive the Bellman expectation equation for V^π.</summary>

**Answer**: Starting from V^π(s) = E[G_t | S_t = s]...

**Explanation**: The key insight is applying the recursive return formula G_t = R_{t+1} + γG_{t+1}, then expanding the expectation over policy and transitions.

**Key equation**: V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]

**Common pitfall**: Forgetting to sum over both actions AND next states. The policy gives action probabilities, the dynamics give state probabilities.
</details>
```

## References

- MkDocs Material documentation: https://squidfunk.github.io/mkdocs-material/
- MkDocs i18n plugin: https://github.com/ultrabug/mkdocs-static-i18n
- Sutton & Barto for RL content structure
- Technical writing best practices for interview prep materials
