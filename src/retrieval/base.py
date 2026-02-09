"""
Base classes and data models for modular retrieval system.

This module defines the core abstractions that enable experimentation
with different retrieval strategies (dense, sparse, hybrid) while
maintaining a consistent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RetrievalStrategy(Enum):
    """Enumeration of available retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


@dataclass
class SearchConfig:
    """
    Configuration for search behavior.

    Allows fine-grained control over retrieval parameters for experimentation.
    """
    top_k: int = 10
    min_score: float = 0.0

    # Dense search specific
    dense_weight: float = 0.5
    dense_top_k: int = 50  # How many to retrieve before fusion

    # Sparse search specific
    sparse_weight: float = 0.5
    sparse_top_k: int = 50

    # Fusion parameters
    rrf_k: int = 60  # RRF constant (research-backed default)

    # Metadata filters (future)
    date_range: Optional[tuple] = None
    keywords_filter: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")
        if not 0 <= self.sparse_weight <= 1:
            raise ValueError("sparse_weight must be between 0 and 1")


@dataclass
class RetrievalResult:
    """
    Standardized result format for all retrieval methods.

    This common format enables fair comparison between different strategies.
    """
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

    # Provenance tracking (for analysis)
    retrieval_method: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Original rank from each method (for fusion analysis)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None

    def __lt__(self, other):
        """Enable sorting by score."""
        return self.score > other.score  # Higher scores first


@dataclass
class RetrievalResponse:
    """
    Complete response from a retrieval operation.

    Includes results plus metadata for analysis and debugging.
    """
    results: List[RetrievalResult]
    query: str
    strategy: RetrievalStrategy
    config: SearchConfig

    # Performance metrics
    total_time_ms: float = 0.0
    dense_time_ms: float = 0.0
    sparse_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0

    # Statistics
    total_candidates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export for analysis/comparison."""
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "config": {
                "top_k": self.config.top_k,
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
                "rrf_k": self.config.rrf_k,
            },
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "metadata": r.metadata,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "dense_rank": r.dense_rank,
                    "sparse_rank": r.sparse_rank,
                }
                for r in self.results
            ],
            "performance": {
                "total_time_ms": self.total_time_ms,
                "dense_time_ms": self.dense_time_ms,
                "sparse_time_ms": self.sparse_time_ms,
                "fusion_time_ms": self.fusion_time_ms,
            },
            "total_candidates": self.total_candidates,
        }


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods.

    Enforces a consistent interface while allowing different implementations
    (dense vector search, BM25, hybrid, reranked, etc.)
    """

    @abstractmethod
    def search(self, query: str, config: SearchConfig) -> List[RetrievalResult]:
        """
        Execute search and return results.

        Args:
            query: User query string
            config: Search configuration parameters

        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> RetrievalStrategy:
        """Return the strategy this retriever implements."""
        pass

    def search_with_timing(self, query: str, config: SearchConfig) -> RetrievalResponse:
        """
        Execute search with performance tracking.

        Convenience method that wraps search() with timing information.
        """
        import time

        start = time.time()
        results = self.search(query, config)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        return RetrievalResponse(
            results=results,
            query=query,
            strategy=self.get_strategy_name(),
            config=config,
            total_time_ms=elapsed,
            total_candidates=len(results),
        )


class BaseFusion(ABC):
    """
    Abstract base class for result fusion strategies.

    Enables experimentation with different fusion methods (RRF, weighted, etc.)
    """

    @abstractmethod
    def fuse(
        self,
        results_lists: List[List[RetrievalResult]],
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """
        Combine multiple ranked lists into a single ranking.

        Args:
            results_lists: List of ranked result lists from different retrievers
            config: Configuration (may include fusion parameters)

        Returns:
            Fused ranking as List[RetrievalResult]
        """
        pass

    @abstractmethod
    def get_fusion_name(self) -> str:
        """Return the name of this fusion strategy."""
        pass
