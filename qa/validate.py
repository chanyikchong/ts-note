#!/usr/bin/env python3
"""
Documentation Validator for Time Series Study Notes

Validates:
- EN and ZH file trees match
- Required sections exist in each page
- Quiz sections have >= 5 questions
- Internal links are valid
- Markdown syntax is valid

Usage:
    python -m qa.validate
    python -m qa.validate --errors-only
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationError:
    """Represents a validation error."""
    file: str
    error_type: str
    message: str
    line: int = 0
    severity: str = "error"  # error, warning

    def __str__(self):
        loc = f":{self.line}" if self.line else ""
        return f"[{self.severity.upper()}] {self.file}{loc}: {self.error_type} - {self.message}"


class DocumentValidator:
    """Validates Time Series Study Notes documentation."""

    REQUIRED_SECTIONS = [
        "面试摘要|Interview Summary|interview-summary",
        "核心定义|Core Definitions",
        "数学与推导|Math and Derivations",
        "算法|Algorithm|Model",
        "常见陷阱|Common Pitfalls",
        "简单示例|Mini Example|Example",
        "测验|Quiz",
        "参考文献|References"
    ]

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.errors: List[ValidationError] = []

    def validate_all(self) -> List[ValidationError]:
        """Run all validations."""
        self.errors = []

        self._validate_tree_match()
        self._validate_sections()
        self._validate_quizzes()
        self._validate_links()
        self._validate_markdown()

        return self.errors

    def _get_relative_paths(self, lang: str) -> Set[str]:
        """Get all markdown file paths relative to language directory."""
        lang_dir = self.docs_dir / lang
        if not lang_dir.exists():
            return set()

        paths = set()
        for md_file in lang_dir.rglob("*.md"):
            rel_path = md_file.relative_to(lang_dir)
            paths.add(str(rel_path))

        return paths

    def _validate_tree_match(self):
        """Ensure EN and ZH have matching file trees."""
        en_paths = self._get_relative_paths("en")
        zh_paths = self._get_relative_paths("zh")

        # Files in EN but not ZH
        for path in en_paths - zh_paths:
            self.errors.append(ValidationError(
                file=f"docs/zh/{path}",
                error_type="missing_file",
                message=f"Chinese translation missing (exists in EN)",
                severity="error"
            ))

        # Files in ZH but not EN
        for path in zh_paths - en_paths:
            self.errors.append(ValidationError(
                file=f"docs/en/{path}",
                error_type="missing_file",
                message=f"English version missing (exists in ZH)",
                severity="warning"
            ))

    def _validate_sections(self):
        """Validate required sections exist in each page."""
        for lang in ["en", "zh"]:
            lang_dir = self.docs_dir / lang
            if not lang_dir.exists():
                continue

            for md_file in lang_dir.rglob("*.md"):
                # Skip index files and interview question compilations
                if md_file.name in ("index.md", "interview-questions.md"):
                    continue

                content = md_file.read_text(encoding='utf-8')
                rel_path = str(md_file.relative_to(self.docs_dir))

                for section_pattern in self.REQUIRED_SECTIONS:
                    patterns = section_pattern.split("|")
                    found = False

                    for pattern in patterns:
                        # Check for ## heading or HTML class
                        if re.search(rf'^##\s+.*{re.escape(pattern)}', content, re.MULTILINE | re.IGNORECASE):
                            found = True
                            break
                        if re.search(rf'class=.*{re.escape(pattern)}', content, re.IGNORECASE):
                            found = True
                            break

                    if not found:
                        self.errors.append(ValidationError(
                            file=rel_path,
                            error_type="missing_section",
                            message=f"Missing required section: {section_pattern}",
                            severity="warning"
                        ))

    def _validate_quizzes(self):
        """Validate quiz sections have >= 5 questions and EN/ZH counts match."""
        quiz_counts: Dict[str, Dict[str, int]] = defaultdict(dict)

        for lang in ["en", "zh"]:
            lang_dir = self.docs_dir / lang
            if not lang_dir.exists():
                continue

            for md_file in lang_dir.rglob("*.md"):
                if md_file.name == "index.md":
                    continue

                content = md_file.read_text(encoding='utf-8')
                rel_path = str(md_file.relative_to(lang_dir))
                full_path = str(md_file.relative_to(self.docs_dir))

                # Count <details> elements (quiz questions)
                details_count = len(re.findall(r'<details', content, re.IGNORECASE))

                quiz_counts[rel_path][lang] = details_count

                # Check minimum count (skip interview-questions which may have different format)
                if details_count < 5 and "interview" not in rel_path:
                    self.errors.append(ValidationError(
                        file=full_path,
                        error_type="insufficient_quiz",
                        message=f"Quiz has only {details_count} questions (minimum 5 required)",
                        severity="error"
                    ))

        # Check EN/ZH quiz counts match
        for rel_path, counts in quiz_counts.items():
            en_count = counts.get("en", 0)
            zh_count = counts.get("zh", 0)

            if en_count != zh_count and en_count > 0 and zh_count > 0:
                self.errors.append(ValidationError(
                    file=f"docs/{{en,zh}}/{rel_path}",
                    error_type="quiz_count_mismatch",
                    message=f"Quiz count mismatch: EN={en_count}, ZH={zh_count}",
                    severity="error"
                ))

    def _validate_links(self):
        """Validate internal links point to existing files."""
        for lang in ["en", "zh"]:
            lang_dir = self.docs_dir / lang
            if not lang_dir.exists():
                continue

            for md_file in lang_dir.rglob("*.md"):
                content = md_file.read_text(encoding='utf-8')
                rel_path = str(md_file.relative_to(self.docs_dir))

                # Find markdown links
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

                for link_text, link_url in links:
                    # Skip external links
                    if link_url.startswith(('http://', 'https://', '#', 'mailto:')):
                        continue

                    # Resolve relative link
                    if link_url.startswith('/'):
                        target = self.docs_dir / link_url.lstrip('/')
                    else:
                        target = md_file.parent / link_url

                    # Remove anchor
                    target_str = str(target).split('#')[0]
                    target = Path(target_str)

                    # Check if file exists
                    if not target.exists() and not Path(target_str + ".md").exists():
                        self.errors.append(ValidationError(
                            file=rel_path,
                            error_type="broken_link",
                            message=f"Broken link: {link_url}",
                            severity="warning"
                        ))

    def _validate_markdown(self):
        """Basic markdown syntax validation."""
        for lang in ["en", "zh"]:
            lang_dir = self.docs_dir / lang
            if not lang_dir.exists():
                continue

            for md_file in lang_dir.rglob("*.md"):
                content = md_file.read_text(encoding='utf-8')
                rel_path = str(md_file.relative_to(self.docs_dir))
                lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    # Check for unclosed code blocks
                    if line.strip().startswith('```') and line.strip() != '```':
                        code_lang = line.strip()[3:]
                        # Look for closing
                        found_close = False
                        for j in range(i, len(lines)):
                            if lines[j].strip() == '```':
                                found_close = True
                                break
                        if not found_close:
                            self.errors.append(ValidationError(
                                file=rel_path,
                                error_type="unclosed_code_block",
                                message=f"Unclosed code block",
                                line=i,
                                severity="warning"
                            ))

                    # Check for unclosed details tags
                    if '<details>' in line.lower() or '<details ' in line.lower():
                        remaining = '\n'.join(lines[i-1:])
                        open_count = len(re.findall(r'<details', remaining, re.IGNORECASE))
                        close_count = len(re.findall(r'</details>', remaining, re.IGNORECASE))
                        if open_count > close_count:
                            self.errors.append(ValidationError(
                                file=rel_path,
                                error_type="unclosed_details",
                                message="Unclosed <details> tag",
                                line=i,
                                severity="warning"
                            ))

    def print_report(self):
        """Print validation report."""
        if not self.errors:
            print("All validations passed!")
            return

        # Group by file
        by_file = defaultdict(list)
        for error in self.errors:
            by_file[error.file].append(error)

        error_count = sum(1 for e in self.errors if e.severity == "error")
        warning_count = sum(1 for e in self.errors if e.severity == "warning")

        print(f"\n{'='*60}")
        print(f"Validation Report: {error_count} errors, {warning_count} warnings")
        print(f"{'='*60}\n")

        for file, file_errors in sorted(by_file.items()):
            print(f"FILE: {file}")
            for error in file_errors:
                icon = "[ERROR]" if error.severity == "error" else "[WARN]"
                line_info = f" (line {error.line})" if error.line else ""
                print(f"   {icon} {error.error_type}{line_info}: {error.message}")
            print()

        print(f"{'='*60}")
        print(f"Total: {error_count} errors, {warning_count} warnings")

        if error_count > 0:
            print("\nValidation FAILED")
            sys.exit(1)
        else:
            print("\nValidation passed with warnings")


def main():
    parser = argparse.ArgumentParser(description="Validate Time Series Study Notes documentation")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--errors-only", action="store_true", help="Only show errors, not warnings")
    args = parser.parse_args()

    validator = DocumentValidator(args.docs_dir)
    errors = validator.validate_all()

    if args.errors_only:
        errors = [e for e in errors if e.severity == "error"]
        validator.errors = errors

    if args.json:
        import json
        print(json.dumps([{
            "file": e.file,
            "type": e.error_type,
            "message": e.message,
            "line": e.line,
            "severity": e.severity
        } for e in errors], indent=2))
    else:
        validator.print_report()


if __name__ == "__main__":
    main()
