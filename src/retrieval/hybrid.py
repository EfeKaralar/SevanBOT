"""
Hybrid retrieval combining dense and sparse search.

This module implements hybrid search by fusing results from multiple
retrieval strategies using configurable fusion methods.
"""

import time
from typing import List

from .base import (
    BaseRetriever,
    BaseFusion,
    RetrievalResult,
    SearchConfig,
    RetrievalStrategy,
    RetrievalResponse
)
from .dense import DenseRetriever
from .sparse import SparseRetriever
from .fusion import RRFFusion


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining multiple retrieval strategies.

    Executes dense and sparse search in parallel, then fuses results
    using a configurable fusion strategy (RRF by default).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_strategy: BaseFusion = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Dense vector search retriever
            sparse_retriever: Sparse BM25 retriever
            fusion_strategy: Fusion method (defaults to RRF)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_strategy = fusion_strategy or RRFFusion()

        print(f"[HYBRID] Using fusion strategy: {self.fusion_strategy.get_fusion_name()}")

    def search(self, query: str, config: SearchConfig) -> List[RetrievalResult]:
        """
        Execute hybrid search.

        Runs dense and sparse search, then fuses results.

        Args:
            query: User query string
            config: Search configuration

        Returns:
            Fused ranking of results
        """
        # Execute both searches
        dense_results = self.dense_retriever.search(query, config)
        sparse_results = self.sparse_retriever.search(query, config)

        # Fuse results
        fused_results = self.fusion_strategy.fuse(
            [dense_results, sparse_results],
            config
        )

        # Return top-k from fused results
        return fused_results[:config.top_k]

    def search_with_timing(self, query: str, config: SearchConfig) -> RetrievalResponse:
        """
        Execute hybrid search with detailed timing breakdown.

        Overrides base method to track dense, sparse, and fusion times separately.
        """
        # Time dense search
        start = time.time()
        dense_results = self.dense_retriever.search(query, config)
        dense_time = (time.time() - start) * 1000

        # Time sparse search
        start = time.time()
        sparse_results = self.sparse_retriever.search(query, config)
        sparse_time = (time.time() - start) * 1000

        # Time fusion
        start = time.time()
        fused_results = self.fusion_strategy.fuse(
            [dense_results, sparse_results],
            config
        )
        fusion_time = (time.time() - start) * 1000

        # Get top-k
        final_results = fused_results[:config.top_k]

        # Build response with detailed timing
        return RetrievalResponse(
            results=final_results,
            query=query,
            strategy=self.get_strategy_name(),
            config=config,
            total_time_ms=dense_time + sparse_time + fusion_time,
            dense_time_ms=dense_time,
            sparse_time_ms=sparse_time,
            fusion_time_ms=fusion_time,
            total_candidates=len(dense_results) + len(sparse_results),
        )

    def get_strategy_name(self) -> RetrievalStrategy:
        """Return the strategy this retriever implements."""
        return RetrievalStrategy.HYBRID

    def get_info(self) -> dict:
        """Get information about the hybrid retriever."""
        return {
            "strategy": "hybrid",
            "fusion": self.fusion_strategy.get_fusion_name(),
            "dense_info": self.dense_retriever.get_collection_info(),
            "sparse_info": self.sparse_retriever.get_index_stats(),
        }
