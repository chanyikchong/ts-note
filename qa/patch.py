"""
Patch Proposal Module
=====================

Generates content patches based on Q&A analysis.

Usage:
    from qa.patch import PatchGenerator
    generator = PatchGenerator()
    patch = generator.create_patch(file_path, section, new_content)
    generator.save_patch(patch, 'clarify-arima-selection')
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import difflib


class PatchGenerator:
    """
    Generates patch proposals for documentation updates.
    """

    def __init__(self, proposals_dir: str = 'proposals'):
        self.proposals_dir = Path(proposals_dir)
        self.proposals_dir.mkdir(parents=True, exist_ok=True)

    def create_unified_diff(self, original: str, modified: str, file_path: str) -> str:
        """Create a unified diff between original and modified content."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            lineterm=''
        )

        return ''.join(diff)

    def find_section(self, content: str, section_heading: str) -> Optional[Dict]:
        """Find a section by heading and return its bounds."""
        lines = content.split('\n')
        section_start = None
        section_end = None
        section_level = None

        for i, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                if section_start is not None:
                    # Check if this heading ends our section
                    if level <= section_level:
                        section_end = i
                        break
                elif heading.lower() == section_heading.lower():
                    section_start = i
                    section_level = level

        if section_start is not None and section_end is None:
            section_end = len(lines)

        if section_start is not None:
            return {
                'start': section_start,
                'end': section_end,
                'content': '\n'.join(lines[section_start:section_end])
            }

        return None

    def create_section_patch(
        self,
        file_path: str,
        section_heading: str,
        new_section_content: str,
        reason: str = ""
    ) -> Dict:
        """
        Create a patch to update a specific section.

        Parameters:
        -----------
        file_path : str - Path to the markdown file
        section_heading : str - Heading of section to update
        new_section_content : str - New content for the section
        reason : str - Reason for the change

        Returns:
        --------
        Patch dictionary with metadata and diff
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        original_content = path.read_text(encoding='utf-8')
        section = self.find_section(original_content, section_heading)

        if section is None:
            # Section doesn't exist - add it at the end
            modified_content = f"{original_content}\n\n{new_section_content}"
        else:
            # Replace existing section
            lines = original_content.split('\n')
            modified_lines = (
                lines[:section['start']] +
                new_section_content.split('\n') +
                lines[section['end']:]
            )
            modified_content = '\n'.join(modified_lines)

        diff = self.create_unified_diff(original_content, modified_content, str(path))

        return {
            'file_path': str(path),
            'section': section_heading,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'diff': diff,
            'original_content': section['content'] if section else None,
            'new_content': new_section_content
        }

    def create_addition_patch(
        self,
        file_path: str,
        content_to_add: str,
        after_section: Optional[str] = None,
        reason: str = ""
    ) -> Dict:
        """
        Create a patch to add new content.

        Parameters:
        -----------
        file_path : str - Path to the markdown file
        content_to_add : str - Content to add
        after_section : str - Add after this section (or at end if None)
        reason : str - Reason for the addition

        Returns:
        --------
        Patch dictionary
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        original_content = path.read_text(encoding='utf-8')

        if after_section:
            section = self.find_section(original_content, after_section)
            if section:
                lines = original_content.split('\n')
                insert_point = section['end']
                modified_lines = (
                    lines[:insert_point] +
                    [''] + content_to_add.split('\n') +
                    lines[insert_point:]
                )
                modified_content = '\n'.join(modified_lines)
            else:
                # Section not found, add at end
                modified_content = f"{original_content}\n\n{content_to_add}"
        else:
            modified_content = f"{original_content}\n\n{content_to_add}"

        diff = self.create_unified_diff(original_content, modified_content, str(path))

        return {
            'file_path': str(path),
            'section': f"New content after {after_section}" if after_section else "End of file",
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'diff': diff,
            'new_content': content_to_add
        }

    def create_quiz_patch(
        self,
        file_path: str,
        questions: List[Dict],
        reason: str = "Add quiz questions based on user confusion"
    ) -> Dict:
        """
        Create a patch to add quiz questions.

        Parameters:
        -----------
        file_path : str - Path to the markdown file
        questions : List[Dict] - List of {question, answer, explanation}
        reason : str - Reason for adding

        Returns:
        --------
        Patch dictionary
        """
        # Format quiz questions
        quiz_content = []
        for q in questions:
            quiz_content.append(f"""
<details>
<summary>{q['question']}</summary>

**Answer:** {q['answer']}

**Explanation:** {q['explanation']}

</details>
""".strip())

        content = "\n\n".join(quiz_content)

        return self.create_section_patch(
            file_path,
            "Quiz",
            f"## Quiz\n\n{content}",
            reason
        )

    def save_patch(self, patch: Dict, name: str) -> str:
        """
        Save a patch proposal to the proposals directory.

        Parameters:
        -----------
        patch : Dict - Patch dictionary
        name : str - Name for the patch file

        Returns:
        --------
        Path to saved patch file
        """
        # Sanitize name
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{safe_name}.patch.md"

        patch_path = self.proposals_dir / filename

        # Format patch as markdown
        content = f"""# Patch Proposal: {name}

**File:** `{patch['file_path']}`
**Section:** {patch['section']}
**Created:** {patch['timestamp']}

## Reason

{patch['reason']}

## Changes

```diff
{patch['diff']}
```

## New Content

```markdown
{patch.get('new_content', 'N/A')}
```

## How to Apply

1. Review the changes above
2. If approved, manually apply the diff or copy the new content
3. Run `python -m qa.validate` to verify the changes

---
*Generated by QA system*
"""

        patch_path.write_text(content, encoding='utf-8')
        print(f"Patch saved to: {patch_path}")

        return str(patch_path)

    def save_to_claude_temp(self, patch: Dict, name: str) -> str:
        """Save patch to .claude/ temporary directory first."""
        claude_dir = Path('.claude/patches')
        claude_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{safe_name}.patch.json"

        import json
        patch_path = claude_dir / filename
        with open(patch_path, 'w', encoding='utf-8') as f:
            json.dump(patch, f, indent=2, ensure_ascii=False)

        return str(patch_path)


def main():
    """Demo patch generation."""
    generator = PatchGenerator()

    # Example: Create a quiz patch
    example_questions = [
        {
            'question': 'What is the key assumption of ARIMA models?',
            'answer': 'The differenced series must be stationary.',
            'explanation': 'ARIMA applies differencing d times to achieve stationarity, '
                          'then fits an ARMA model to the result.'
        },
        {
            'question': 'When should you use SARIMA over ARIMA?',
            'answer': 'When the data exhibits seasonal patterns.',
            'explanation': 'SARIMA extends ARIMA with seasonal components (P, D, Q, s) '
                          'to capture periodic patterns in the data.'
        }
    ]

    print("Patch Generator Demo")
    print("=" * 50)
    print("\nExample quiz patch format:")
    print("-" * 50)

    for q in example_questions:
        print(f"""
<details>
<summary>{q['question']}</summary>

**Answer:** {q['answer']}

**Explanation:** {q['explanation']}

</details>
""")


if __name__ == '__main__':
    main()
