"""
Sparse retrieval using BM25 keyword search.

This module implements traditional keyword-based search using BM25 algorithm
with optional Turkish stemming for better recall.
"""

import json
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import bm25s
    import Stemmer  # PyStemmer
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False
    print("[WARNING] bm25s or PyStemmer not available. Install with: pip install bm25s PyStemmer")

from .base import BaseRetriever, RetrievalResult, SearchConfig, RetrievalStrategy


class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using BM25 keyword matching.

    Provides traditional keyword-based search as a complement to dense retrieval.
    Supports Turkish stemming for better morphological matching.
    """

    def __init__(
        self,
        chunks_file: str,
        use_stemming: bool = True,
        language: str = "turkish",
    ):
        """
        Initialize sparse retriever with BM25 index.

        Args:
            chunks_file: Path to JSONL file with chunks
            use_stemming: Whether to apply stemming (improves Turkish recall)
            language: Language for stemming (default: turkish)
        """
        if not STEMMER_AVAILABLE:
            raise ImportError(
                "BM25 dependencies not available. Install with:\n"
                "  pip install bm25s PyStemmer"
            )

        self.chunks_file = Path(chunks_file)
        self.use_stemming = use_stemming
        self.language = language

        # Load chunks
        self.chunks = self._load_chunks()
        print(f"[SPARSE] Loaded {len(self.chunks)} chunks from {chunks_file}")

        # Initialize stemmer
        self.stemmer = None
        if use_stemming:
            try:
                self.stemmer = Stemmer.Stemmer(language)
                print(f"[SPARSE] Using {language} stemmer")
            except KeyError:
                print(f"[WARNING] Stemmer for '{language}' not available, using no stemming")
                self.use_stemming = False

        # Build BM25 index
        print("[SPARSE] Building BM25 index...")
        self.retriever = self._build_index()
        print("[SPARSE] BM25 index ready")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file."""
        chunks = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize and optionally stem text.

        Args:
            text: Input text

        Returns:
            List of tokens (stemmed if stemming is enabled)
        """
        # Simple tokenization: lowercase + split on non-alphanumeric
        # Keep Turkish characters (ç, ğ, ı, İ, ö, ş, ü)
        tokens = re.findall(r'\w+', text.lower())

        # Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stemWord(token) for token in tokens]

        return tokens

    def _build_index(self) -> Any:
        """Build BM25 index from chunks."""
        # Extract corpus (content for indexing)
        corpus_texts = []
        for chunk in self.chunks:
            # Try different field names (depends on chunking format)
            text_for_embedding = chunk.get("text_for_embedding", "")
            text = chunk.get("text", "")
            simple_context = chunk.get("simple_context", "")
            content = chunk.get("content", "")

            # Use text_for_embedding if available (has context), otherwise combine
            if text_for_embedding:
                full_text = text_for_embedding
            elif content:
                full_text = f"{simple_context} {content}" if simple_context else content
            else:
                full_text = text

            corpus_texts.append(full_text)

        # Tokenize corpus
        corpus_tokens = [self._tokenize(text) for text in corpus_texts]

        # Build BM25 index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        return retriever

    def search(self, query: str, config: SearchConfig) -> List[RetrievalResult]:
        """
        Execute BM25 keyword search.

        Args:
            query: User query string
            config: Search configuration

        Returns:
            List of RetrievalResult objects sorted by BM25 score
        """
        # Tokenize query
        query_tokens = self._tokenize(query)

        # Use sparse_top_k if available, otherwise fall back to top_k
        limit = config.sparse_top_k if hasattr(config, 'sparse_top_k') else config.top_k

        # Search with BM25
        results_array, scores_array = self.retriever.retrieve(
            [query_tokens],  # bm25s expects list of queries
            k=limit
        )

        # Convert to RetrievalResult format
        results = []
        for idx, (doc_idx, score) in enumerate(zip(results_array[0], scores_array[0]), 1):
            # Handle both scalar and array scores from bm25s
            score_value = float(score.item()) if hasattr(score, 'item') else float(score)

            chunk = self.chunks[doc_idx]

            # Extract content (use text if content not available)
            content_text = chunk.get("content", chunk.get("text", ""))

            # Build metadata from chunk fields
            metadata = chunk.get("metadata", {})
            if not metadata:
                # Build metadata from chunk fields if not already present
                metadata = {
                    "title": chunk.get("title", ""),
                    "date": chunk.get("date", ""),
                    "source": chunk.get("source", ""),
                    "keywords": chunk.get("keywords", ""),
                }

            result = RetrievalResult(
                chunk_id=chunk.get("chunk_id", f"chunk_{doc_idx}"),
                score=score_value,
                content=content_text,
                metadata=metadata,
                retrieval_method="sparse",
                sparse_score=score_value,
                sparse_rank=idx,
            )
            results.append(result)

        return results

    def get_strategy_name(self) -> RetrievalStrategy:
        """Return the strategy this retriever implements."""
        return RetrievalStrategy.SPARSE

    def get_index_stats(self) -> dict:
        """Get statistics about the BM25 index."""
        return {
            "total_chunks": len(self.chunks),
            "stemming_enabled": self.use_stemming,
            "language": self.language if self.use_stemming else "none",
        }
