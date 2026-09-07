"""BM25 Keyword Retrieval Engine for Financial Documents.

Implements Okapi BM25 (with positive IDF smoothing) tailored for financial
text and tabular data. Operates directly on `Chunk` models from `src/schemas.py`.
Includes financial acronym expansion, question-word stopword filtering,
and section-aware context injection.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.schemas import Candidate, Chunk, ContentType, RetrievalMethod
except ImportError:
    from schemas import Candidate, Chunk, ContentType, RetrievalMethod


# Question and standard conversational stopwords (filters noise from queries)
DEFAULT_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "had", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to",
    "was", "were", "will", "with",
    # Question words & auxiliary verbs:
    "what", "whats", "how", "much", "many", "did", "do", "does",
    "which", "where", "when", "who", "why", "can", "could", "would", "should",
}


def financial_tokenizer(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text with domain-specific enhancements for financial analysis.
    
    Preserves:
    - Currencies & Magnitudes (e.g., '$120M', '$19 million', '-$5M')
    - Percentages (e.g., '35%', '20%')
    - Years and counts (e.g., '2024', '3000')
    - Hyphenated corporate and regional terms (e.g., 'Asia-Pacific', 'year-over-year')
    - Financial acronym expansions (R&D <-> Research and Development, SG&A, CapEx, OpEx, YoY)
    """
    text_clean = text.lower().replace(",", "").replace("'", "")
    
    extra_tokens: list[str] = []
    
    # -----------------------------------------------------------------------
    # Bidirectional Financial Acronym & Synonym Expansion
    # -----------------------------------------------------------------------
    # R&D <-> Research and Development
    if "r&d" in text_clean or " rd " in f" {text_clean} ":
        extra_tokens.extend(["research", "development", "r&d"])
    if "research and development" in text_clean:
        extra_tokens.extend(["r&d", "rd"])
        
    # SG&A <-> Selling, General and Administrative
    if "sg&a" in text_clean or " sga " in f" {text_clean} ":
        extra_tokens.extend(["selling", "general", "administrative", "sg&a"])
    if "selling general and administrative" in text_clean:
        extra_tokens.extend(["sg&a", "sga"])
        
    # CapEx <-> Capital Expenditures
    if "capex" in text_clean:
        extra_tokens.extend(["capital", "expenditures", "capex"])
    if "capital expenditures" in text_clean or "capital expenditure" in text_clean:
        extra_tokens.extend(["capex"])
        
    # OpEx <-> Operating Expenses
    if "opex" in text_clean:
        extra_tokens.extend(["operating", "expenses", "opex"])
    if "operating expenses" in text_clean or "operating expense" in text_clean:
        extra_tokens.extend(["opex"])
        
    # YoY <-> Year-over-Year
    if "yoy" in text_clean:
        extra_tokens.extend(["year-over-year", "yoy"])
    if "year-over-year" in text_clean or "year over year" in text_clean:
        extra_tokens.extend(["yoy"])

    # Matches negative/positive numbers with optional currency symbols, units, or alphanumeric/hyphenated words
    raw_tokens = re.findall(
        r"-?[\$€£]?\d+(?:\.\d+)?(?:%|[kmbt])?|[a-z0-9&]+(?:-[a-z0-9&]+)*",
        text_clean
    )
    
    tokens: list[str] = []
    for token in raw_tokens:
        if remove_stopwords and token in DEFAULT_STOPWORDS:
            continue
        
        tokens.append(token)
        
        # Sub-token expansion for currencies/units: '$120m' -> also index '120m' and '120'
        clean = re.sub(r"^[\$€£-]", "", token)
        if clean != token and clean:
            tokens.append(clean)
            bare_num = re.sub(r"[kmbt%]$", "", clean)
            if bare_num and bare_num != clean:
                tokens.append(bare_num)
        
        # Sub-token expansion for hyphenated terms: 'asia-pacific' -> also index 'asia', 'pacific'
        if "-" in token:
            for part in token.split("-"):
                if part and (not remove_stopwords or part not in DEFAULT_STOPWORDS):
                    tokens.append(part)
                    
    return tokens + extra_tokens


class BM25Retriever:
    """Okapi BM25 retriever for structured financial chunks and tables."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}  # term -> list[(doc_idx, tf)]

        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 inverted index with section-aware context injection."""
        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.doc_freqs = {}
        self.inverted_index = {}

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_length = 0
        for doc_idx, chunk in enumerate(self.chunks):
            # Section-aware context injection: prepend section heading to text content
            section_prefix = f"{chunk.section}: " if chunk.section else ""
            text = section_prefix + chunk.to_text()
            
            tokens = financial_tokenizer(text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_length += doc_len

            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, tf))

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

    def _idf(self, term: str) -> float:
        """Calculate smoothed Inverse Document Frequency ensuring non-negative values."""
        df = self.doc_freqs.get(term, 0)
        # BM25+ / Lucene smoothed formula: ln(1 + (N - df + 0.5) / (df + 0.5))
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 30,
        metadata_filter: dict[str, Any] | None = None,
        content_type: ContentType | None = None,
        min_score: float = 0.0,
    ) -> list[Candidate]:
        """Retrieve the top_k matching chunks for a given query.
        
        Args:
            query: Natural language query string.
            top_k: Number of candidates to return (default 30 for reranker over-retrieval).
            metadata_filter: Optional dict of key-value pairs matching chunk.metadata or chunk attributes.
            content_type: Optional filter for ContentType (e.g. TEXT, TABLE).
            min_score: Minimum BM25 score threshold to filter out weak evidence.

        Returns:
            Ranked list of Candidate objects with BM25 scores and ranks.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = financial_tokenizer(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        # Accumulate BM25 scores using inverted index
        for term in set(query_tokens):
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        # Apply metadata, content_type, and min_score filters
        filtered_results: list[tuple[int, float]] = []
        for doc_idx, score in scores.items():
            if score < min_score:
                continue

            chunk = self.chunks[doc_idx]

            if content_type and chunk.content_type != content_type:
                continue

            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    chunk_val = chunk.metadata.get(k) if k in chunk.metadata else getattr(chunk, k, None)
                    if chunk_val != v:
                        match = False
                        break
                if not match:
                    continue

            filtered_results.append((doc_idx, score))

        # Sort descending by BM25 score
        filtered_results.sort(key=lambda item: item[1], reverse=True)

        # Build Candidate models
        candidates: list[Candidate] = []
        for rank, (doc_idx, score) in enumerate(filtered_results[:top_k], start=1):
            chunk = self.chunks[doc_idx]
            candidates.append(
                Candidate(
                    chunk=chunk,
                    score=round(score, 4),
                    retrieval_method=RetrievalMethod.BM25,
                    rank=rank,
                    initial_rank=rank,
                    scores={"bm25": round(score, 4)},
                )
            )

        return candidates

    def search_tables(self, query: str, top_k: int = 5) -> list[Candidate]:
        """Convenience method to retrieve specifically from tabular chunks.
        
        Matches the reasoning agent's required tool: `search_tables(query)`.
        """
        return self.search(query=query, top_k=top_k, content_type=ContentType.TABLE)


# ---------------------------------------------------------------------------
# Quick Verification / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_path = Path(__file__).resolve().parent.parent / "data" / "mock_chunks.json"
    if not mock_path.exists():
        print(f"File not found: {mock_path}")
        exit(1)

    with open(mock_path, "r") as f:
        data = json.load(f)

    chunks = [Chunk.model_validate(item) for item in data]
    print(f"Loaded {len(chunks)} chunks into BM25 retriever.")

    retriever = BM25Retriever(chunks)

    test_queries = [
        "What was the operating income in 2024?",
        "Research and development expense 2024",
        "Asia-Pacific revenue growth rate",
        "Cash from operating activities in 2023",
        "Revenue breakdown by region",
        "Average selling price per unit",  # intentional test (missing data)
        "What's the date that the company was founded ? ",
        "How much  R&D did the company spend in 2023?",  # acronym test
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = retriever.search(q, top_k=3)
        for cand in results:
            print(f"  [Rank {cand.rank} | Score {cand.score}] chunk_id={cand.chunk.chunk_id}, page={cand.chunk.page}, section='{cand.chunk.section}', type={cand.chunk.content_type.value}")
