"""
Modular retrieval system for semantic search.

This package provides a flexible, composable retrieval framework supporting:
- Dense retrieval (vector similarity via Qdrant)
- Sparse retrieval (BM25 keyword matching)
- Hybrid retrieval (combining dense + sparse with RRF or weighted fusion)
- Comparison tools for evaluating different strategies

Example usage:
    from retrieval import DenseRetriever, SparseRetriever, HybridRetriever, SearchConfig
    from retrieval import RetrievalComparator

    # Initialize retrievers
    dense = DenseRetriever(collection_name="contextual_chunks")
    sparse = SparseRetriever(chunks_file="chunks_contextual.jsonl")
    hybrid = HybridRetriever(dense, sparse)

    # Search
    config = SearchConfig(top_k=10, rrf_k=60)
    results = hybrid.search("Türk dili tarihi", config)

    # Compare strategies
    comparator = RetrievalComparator({
        "dense": dense,
        "sparse": sparse,
        "hybrid": hybrid
    })
    comparison = comparator.compare("Türk dili tarihi", config)
    comparison.print_summary()
"""

from .base import (
    BaseRetriever,
    BaseFusion,
    RetrievalStrategy,
    SearchConfig,
    RetrievalResult,
    RetrievalResponse,
)

from .dense import DenseRetriever
from .sparse import SparseRetriever
from .fusion import RRFFusion, WeightedFusion
from .hybrid import HybridRetriever
from .evaluator import RetrievalComparator, ComparisonResult

__all__ = [
    # Base classes
    "BaseRetriever",
    "BaseFusion",
    "RetrievalStrategy",
    "SearchConfig",
    "RetrievalResult",
    "RetrievalResponse",
    # Retrievers
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    # Fusion
    "RRFFusion",
    "WeightedFusion",
    # Evaluation
    "RetrievalComparator",
    "ComparisonResult",
]
