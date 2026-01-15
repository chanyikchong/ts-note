"""
Document Indexing Module
========================

Provides BM25-based indexing for time series documentation.

Usage:
    from qa.index import DocumentIndex
    index = DocumentIndex()
    index.build_index('docs/en')
    results = index.search('ARIMA model selection', top_k=5)
"""

import os
import re
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BM25:
    """
    BM25 ranking algorithm implementation.

    BM25 score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))

    Parameters:
    -----------
    k1 : float - Term frequency saturation parameter (default: 1.5)
    b : float - Length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.doc_count: int = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

    def add_document(self, doc_id: str, tokens: List[str]):
        """Add a document to the index."""
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

        # Count term frequencies
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        # Update inverted index and document frequencies
        for term, freq in term_freqs.items():
            if doc_id not in self.inverted_index[term]:
                self.term_doc_freq[term] += 1
            self.inverted_index[term][doc_id] = freq

    def finalize(self):
        """Compute average document length after all documents are added."""
        if self.doc_count > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.doc_count

    def idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = self.term_doc_freq.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_id: str) -> float:
        """Compute BM25 score for a document given a query."""
        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            return 0

        score = 0.0
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            if doc_id not in self.inverted_index[term]:
                continue

            tf = self.inverted_index[term][doc_id]
            idf = self.idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query."""
        # Get candidate documents (those containing at least one query term)
        candidates = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidates.update(self.inverted_index[term].keys())

        # Score all candidates
        scores = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in candidates]

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class DocumentIndex:
    """
    Document index for time series notes.

    Provides search functionality over markdown documentation.
    """

    def __init__(self):
        self.bm25 = BM25()
        self.documents: Dict[str, Dict] = {}  # doc_id -> {path, title, content, sections}
        self.sections: Dict[str, Dict] = {}   # section_id -> {doc_id, heading, content, line_start}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase, alphanumeric, remove stopwords."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'us', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his',
            'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'then', 'if'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"

    def extract_sections(self, content: str, doc_id: str) -> List[Dict]:
        """Extract sections from markdown content."""
        sections = []
        lines = content.split('\n')
        current_heading = None
        current_content = []
        current_line_start = 0

        for i, line in enumerate(lines):
            # Check for heading
            heading_match = re.match(r'^(#{1,3})\s+(.+)$', line)
            if heading_match:
                # Save previous section
                if current_heading is not None:
                    section_content = '\n'.join(current_content).strip()
                    if section_content:
                        sections.append({
                            'heading': current_heading,
                            'content': section_content,
                            'line_start': current_line_start,
                            'level': len(heading_match.group(1))
                        })

                current_heading = heading_match.group(2).strip()
                current_content = []
                current_line_start = i + 1
            else:
                current_content.append(line)

        # Save last section
        if current_heading is not None:
            section_content = '\n'.join(current_content).strip()
            if section_content:
                sections.append({
                    'heading': current_heading,
                    'content': section_content,
                    'line_start': current_line_start,
                    'level': 1
                })

        return sections

    def add_document(self, file_path: Path, content: str):
        """Add a document to the index."""
        doc_id = str(file_path)
        title = self.extract_title(content)
        sections = self.extract_sections(content, doc_id)

        # Store document metadata
        self.documents[doc_id] = {
            'path': str(file_path),
            'title': title,
            'content': content,
            'sections': sections
        }

        # Index document-level content
        doc_tokens = self.tokenize(f"{title} {content}")
        self.bm25.add_document(doc_id, doc_tokens)

        # Index sections
        for i, section in enumerate(sections):
            section_id = f"{doc_id}#section_{i}"
            section_tokens = self.tokenize(f"{section['heading']} {section['content']}")
            self.bm25.add_document(section_id, section_tokens)

            self.sections[section_id] = {
                'doc_id': doc_id,
                'heading': section['heading'],
                'content': section['content'],
                'line_start': section['line_start']
            }

    def build_index(self, docs_path: str, language: str = 'en'):
        """Build index from markdown files in the given path."""
        docs_dir = Path(docs_path) / language
        if not docs_dir.exists():
            docs_dir = Path(docs_path)

        md_files = list(docs_dir.rglob('*.md'))

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
                self.add_document(md_file, content)
            except Exception as e:
                print(f"Warning: Failed to index {md_file}: {e}")

        self.bm25.finalize()
        print(f"Indexed {len(self.documents)} documents and {len(self.sections)} sections")

    def search(self, query: str, top_k: int = 5, include_sections: bool = True) -> List[Dict]:
        """
        Search the index for relevant documents/sections.

        Parameters:
        -----------
        query : str - Search query
        top_k : int - Number of results to return
        include_sections : bool - If True, search sections; if False, only documents

        Returns:
        --------
        List of result dictionaries with path, title, content, score
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        results = self.bm25.search(query_tokens, top_k=top_k * 2)  # Get more to filter

        output = []
        for doc_id, score in results:
            if score <= 0:
                continue

            if doc_id in self.documents:
                # Full document result
                doc = self.documents[doc_id]
                output.append({
                    'type': 'document',
                    'path': doc['path'],
                    'title': doc['title'],
                    'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
                    'score': score
                })
            elif include_sections and doc_id in self.sections:
                # Section result
                section = self.sections[doc_id]
                parent_doc = self.documents.get(section['doc_id'], {})
                output.append({
                    'type': 'section',
                    'path': parent_doc.get('path', ''),
                    'title': f"{parent_doc.get('title', '')} > {section['heading']}",
                    'heading': section['heading'],
                    'content': section['content'][:500] + '...' if len(section['content']) > 500 else section['content'],
                    'line_start': section['line_start'],
                    'score': score
                })

            if len(output) >= top_k:
                break

        return output

    def save_index(self, path: str = '.claude/index.json'):
        """Save index to file for faster loading."""
        index_data = {
            'documents': self.documents,
            'sections': self.sections,
            'bm25': {
                'doc_lengths': self.bm25.doc_lengths,
                'avg_doc_length': self.bm25.avg_doc_length,
                'doc_count': self.bm25.doc_count,
                'term_doc_freq': dict(self.bm25.term_doc_freq),
                'inverted_index': {k: dict(v) for k, v in self.bm25.inverted_index.items()}
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False)
        print(f"Index saved to {path}")

    def load_index(self, path: str = '.claude/index.json') -> bool:
        """Load index from file."""
        if not Path(path).exists():
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            self.documents = index_data['documents']
            self.sections = index_data['sections']

            bm25_data = index_data['bm25']
            self.bm25.doc_lengths = bm25_data['doc_lengths']
            self.bm25.avg_doc_length = bm25_data['avg_doc_length']
            self.bm25.doc_count = bm25_data['doc_count']
            self.bm25.term_doc_freq = defaultdict(int, bm25_data['term_doc_freq'])
            self.bm25.inverted_index = defaultdict(dict)
            for term, docs in bm25_data['inverted_index'].items():
                self.bm25.inverted_index[term] = docs

            print(f"Index loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False


def main():
    """CLI for index management."""
    import argparse

    parser = argparse.ArgumentParser(description='Build and manage document index')
    parser.add_argument('--build', action='store_true', help='Build index from docs/')
    parser.add_argument('--language', '-l', default='en', help='Language to index (en or zh)')
    parser.add_argument('--search', '-s', type=str, help='Search query')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')

    args = parser.parse_args()

    index = DocumentIndex()

    if args.build:
        index.build_index('docs', language=args.language)
        index.save_index()
    elif args.search:
        if not index.load_index():
            print("No index found. Building...")
            index.build_index('docs', language=args.language)
            index.save_index()

        results = index.search(args.search, top_k=args.top_k)
        print(f"\nSearch results for: '{args.search}'")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['type']}] {result['title']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Path: {result['path']}")
            print(f"   Preview: {result['content'][:200]}...")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
