"""
Dense retrieval using vector similarity search via Qdrant.

This module implements semantic search using pre-computed embeddings.
"""

import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

from .base import BaseRetriever, RetrievalResult, SearchConfig, RetrievalStrategy


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using vector similarity search.

    Uses Qdrant for efficient vector search and OpenAI for query embedding.
    """

    def __init__(
        self,
        collection_name: str = "sevanbot_openai-small",
        qdrant_url: Optional[str] = None,
        qdrant_path: str = ".qdrant",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize dense retriever.

        Args:
            collection_name: Qdrant collection to search
            qdrant_url: Qdrant server URL (None for local storage)
            qdrant_path: Local storage path (used if qdrant_url is None)
            embedding_model: OpenAI embedding model name
            openai_api_key: OpenAI API key (reads from env if None)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize Qdrant client (local or remote)
        if qdrant_url:
            print(f"[DENSE] Connecting to Qdrant at {qdrant_url}")
            self.qdrant_client = QdrantClient(url=qdrant_url)
        else:
            print(f"[DENSE] Using local Qdrant storage: {qdrant_path}")
            self.qdrant_client = QdrantClient(path=qdrant_path)

        # Verify collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            raise ValueError(
                f"Collection '{collection_name}' not found. "
                f"Available: {collection_names}"
            )

        # Initialize OpenAI client for query embedding
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set in environment")
        self.openai_client = OpenAI(api_key=api_key)

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query string."""
        response = self.openai_client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def search(self, query: str, config: SearchConfig) -> List[RetrievalResult]:
        """
        Execute dense vector search.

        Args:
            query: User query string
            config: Search configuration

        Returns:
            List of RetrievalResult objects sorted by similarity score
        """
        # Embed the query
        query_vector = self._embed_query(query)

        # Use dense_top_k if available, otherwise fall back to top_k
        limit = config.dense_top_k if hasattr(config, 'dense_top_k') else config.top_k

        # Search Qdrant (using query_points for newer API)
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=config.min_score,
        ).points

        # Convert to RetrievalResult format
        results = []
        for idx, hit in enumerate(search_result, 1):
            # Extract content (use 'text' field from payload)
            content_text = hit.payload.get("text", hit.payload.get("content", ""))

            # Build metadata from payload fields
            metadata = {
                "title": hit.payload.get("title", ""),
                "date": hit.payload.get("date", ""),
                "source": hit.payload.get("source", ""),
                "keywords": hit.payload.get("keywords", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "total_chunks": hit.payload.get("total_chunks", 0),
            }

            result = RetrievalResult(
                chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                score=hit.score,
                content=content_text,
                metadata=metadata,
                retrieval_method="dense",
                dense_score=hit.score,
                dense_rank=idx,
            )
            results.append(result)

        return results

    def get_strategy_name(self) -> RetrievalStrategy:
        """Return the strategy this retriever implements."""
        return RetrievalStrategy.DENSE

    def get_collection_info(self) -> dict:
        """Get information about the collection (for debugging)."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
        }
