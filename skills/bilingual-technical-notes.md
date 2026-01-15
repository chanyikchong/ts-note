---
name: "Bilingual Technical Notes with Math + Code + Quizzes"
description: "Create individual bilingual technical documentation pages with mathematical rigor, code examples, and interactive quizzes. Use when writing technical notes that need: (1) EN/ZH parallel content, (2) LaTeX equations with explanations, (3) runnable code snippets, (4) self-assessment quizzes, or (5) interview-ready format. Ideal for ML/AI, algorithms, systems design topics."
---

# Bilingual Technical Notes with Math + Code + Quizzes

## Overview

Write a single technical note page with bilingual content (English and Chinese), mathematical derivations, code examples, and interactive quiz questions. Focuses on clarity, rigor, and interview-readiness.

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| Topic | Yes | Specific technical concept to document |
| Prerequisites | No | Topics the reader should already know |
| Code language | No | Python (default), or other language |
| Target audience | No | Interview prep (default), research, or learning |

## Outputs

| Output | Format | Description |
|--------|--------|-------------|
| EN page | Markdown | English documentation page |
| ZH page | Markdown | Chinese translation (mirrored structure) |
| Code snippet | Python/other | Embedded runnable example |

## Procedure

### Step 1: Identify Core Concepts

Before writing, answer:
- What is the ONE key insight of this topic?
- What equation/algorithm MUST readers memorize?
- What's the most common mistake?
- How does this connect to other topics?

### Step 2: Write Interview Summary (EN)

**Format:**
```markdown
## Interview Summary

[2-3 sentences: What is this? Why does it matter?]

[1-2 sentences: Key insight or relationship to other concepts]

**What to memorize**: [Key equation, algorithm, or fact in one line]
```

**Guidelines:**
- Assume reader has 30 seconds to review before interview
- Focus on "what would I ask about this?"
- Include the one equation/definition they MUST know

### Step 3: Write Core Definitions

**Format:**
```markdown
## Core Definitions

### [Concept Name]

$$[LaTeX equation]$$

**Meaning**: [One sentence explaining what this represents]

| Term | Symbol | Description |
|------|--------|-------------|
| [term] | [symbol] | [meaning] |
```

**Guidelines:**
- One definition per subsection
- Every equation gets a "Meaning" line
- Use tables for related notation

### Step 4: Write Math and Derivations

**Format:**
```markdown
## Math and Derivations

### [Derivation Title]

Starting from:
$$[starting equation]$$

[Step 1 explanation]
$$[intermediate step]$$

[Step 2 explanation]
$$[next step]$$

**Result:**
$$[final equation]$$
```

**Guidelines:**
- Show key steps, not every algebraic manipulation
- Explain the "why" of each step
- Highlight non-obvious transitions
- Box or bold the final result

### Step 5: Write Algorithm Sketch

**Format:**
```markdown
## Algorithm Sketch

```
Algorithm: [Name]

Input: [inputs]
Output: [outputs]

1. [Step with explanation]
2. [Step with explanation]
   - [Sub-step if needed]
3. [Step with explanation]
```

**Complexity:** Time O(?), Space O(?)
```

**Guidelines:**
- Pseudocode, not language-specific syntax
- Include complexity analysis
- Add inline comments for non-obvious steps

### Step 6: Write Common Pitfalls

**Format:**
```markdown
## Common Pitfalls

1. **[Pitfall name]**: [Brief description]
   - Why it happens: [cause]
   - How to avoid: [solution]

2. **[Pitfall name]**: [Brief description]
   ...
```

**Guidelines:**
- 3-5 pitfalls per topic
- Focus on conceptual errors, not typos
- Include pitfalls from real interviews/debugging

### Step 7: Create Mini Example

**Format:**
```markdown
## Mini Example

**Setup:** [Describe the small problem]

```
[Visual representation if applicable]
```

**Step-by-step:**
1. [First step with values]
2. [Second step with values]
3. [Continue until result]

**Result:** [Final answer with interpretation]
```

**Guidelines:**
- Small enough to trace by hand
- Use concrete numbers
- Show intermediate calculations
- Explain what the result means

### Step 8: Write Quiz Section

Create exactly 5 questions:

**Question types:**
- 2 conceptual (understanding)
- 2 math/derivation (rigor)
- 1 practical/debugging (application)

**Format:**
```html
## Quiz

<details>
<summary><strong>Q1 (Conceptual):</strong> [Question text]</summary>

**Answer**: [Direct answer in 1-2 sentences]

**Explanation**: [Why this is correct, 2-4 sentences]

**Key equation**: [Relevant equation if applicable]
$$[equation]$$

**Common pitfall**: [Mistake people make on this question]
</details>
```

**Guidelines:**
- Questions should test understanding, not recall
- Answers must include explanation
- Math questions require showing key equations
- Practical questions should relate to debugging/implementation

### Step 9: Add References

**Format:**
```markdown
## References

- **[Author]**, [Title], [Chapter/Section if applicable]
- **[Author (Year)]**, [Paper title]
- [Online resource name]: [URL if relevant]

**What to memorize for interviews**: [1-2 sentence summary of essentials]
```

### Step 10: Translate to Chinese

Create parallel ZH version:

**Translation rules:**
1. Same file structure and section order
2. Translate headings:
   - Interview Summary → 面试摘要
   - Core Definitions → 核心定义
   - Math and Derivations → 数学与推导
   - Algorithm Sketch → 算法概述
   - Common Pitfalls → 常见陷阱
   - Mini Example → 小例子
   - Quiz → 测验
   - References → 参考文献
3. Keep LaTeX equations unchanged
4. Keep code in English
5. Use standard Chinese technical terms
6. Keep English acronyms (MDP, TD, CNN)
7. Same number of quiz questions

## Quality Checklist

- [ ] Interview Summary is 3-6 lines and actionable
- [ ] Every equation has a "Meaning" explanation
- [ ] Algorithm includes complexity analysis
- [ ] 3-5 pitfalls with causes and solutions
- [ ] Mini example is traceable by hand
- [ ] Exactly 5 quiz questions (2 conceptual, 2 math, 1 practical)
- [ ] Quiz answers include all 4 components
- [ ] ZH mirrors EN exactly in structure
- [ ] All LaTeX renders correctly
- [ ] Code examples are tested

## Common Failure Modes

| Failure | Cause | Prevention |
|---------|-------|------------|
| Equations without context | Copy-paste from papers | Add "Meaning" line after every equation |
| Quiz too surface-level | Rushing the quiz section | Require math derivation and debugging Qs |
| ZH/EN structure mismatch | Translating content but not structure | Check section-by-section alignment |
| Untested code | Assuming code works | Run every code snippet |
| Interview Summary too detailed | Including everything | Enforce 3-6 line limit |
| Missing practical pitfalls | Only theoretical knowledge | Add at least one debugging/implementation pitfall |

## Examples

### Example 1: Single Equation Definition

**Input:** Define the advantage function

**EN Output:**
```markdown
### Advantage Function

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**Meaning**: How much better is action $a$ compared to the average action under policy $\pi$?

| Component | Symbol | Description |
|-----------|--------|-------------|
| Advantage | $A^\pi(s,a)$ | Relative value of action |
| Action-value | $Q^\pi(s,a)$ | Expected return from (s,a) |
| State-value | $V^\pi(s)$ | Expected return from s |
```

**ZH Output:**
```markdown
### 优势函数

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**含义**：在策略 $\pi$ 下，动作 $a$ 比平均动作好多少？

| 组成部分 | 符号 | 描述 |
|---------|------|-----|
| 优势 | $A^\pi(s,a)$ | 动作的相对价值 |
| 动作价值 | $Q^\pi(s,a)$ | 从(s,a)开始的期望回报 |
| 状态价值 | $V^\pi(s)$ | 从s开始的期望回报 |
```

### Example 2: Quiz Question with Full Answer

**Input:** Create practical quiz question about Q-learning debugging

**Output:**
```html
<details>
<summary><strong>Q5 (Practical):</strong> Your Q-values are growing without bound during training. What's wrong?</summary>

**Answer**: Likely causes: (1) γ=1 with continuing task, (2) positive reward cycles, (3) missing terminal state handling, or (4) learning rate too high.

**Explanation**: Q-values should stabilize around the expected discounted return. Unbounded growth indicates bootstrapping errors are compounding. Most common: forgetting that terminal states should have Q=0 for all actions, or using γ=1 which prevents discounting.

**Key equation**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

If terminal: $r + \gamma \cdot 0$, not $r + \gamma \max Q(s',\cdot)$.

**Common pitfall**: Not checking for `done` flag before computing target. Always set target = r when episode terminates.
</details>
```

## References

- LaTeX/MathJax syntax for equations
- Markdown details/summary for interactive elements
- Technical writing principles for bilingual content
- Interview preparation best practices
