"""
Abstract base class for RAG answer generation.

Mirrors the BaseRetriever pattern from retrieval/base.py, enabling
experimentation with different LLM providers while maintaining
a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

from .config import GenerationConfig
from .response import RAGResponse


class BaseAnswerGenerator(ABC):
    """
    Abstract base class for RAG answer generation.

    Subclasses implement generate() for a specific LLM provider
    (Claude, GPT-4, local models, etc.).
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: GenerationConfig,
        conversation_summary: str | None = None,
        recent_messages: List[Dict[str, str]] | None = None,
        allow_no_sources: bool = False,
    ) -> RAGResponse:
        """
        Generate an answer from a query and retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved chunks, each a dict with keys:
                    text, title, date, chunk_id, score
            config: Generation configuration

        Returns:
            RAGResponse with answer, sources, and usage stats
        """
        pass

    @abstractmethod
    def generate_streaming(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: GenerationConfig,
        conversation_summary: str | None = None,
        recent_messages: List[Dict[str, str]] | None = None,
        allow_no_sources: bool = False,
    ) -> Iterator[str]:
        """
        Generate an answer with streaming (yields tokens as they arrive).

        Args:
            query: User's question
            chunks: Retrieved chunks
            config: Generation configuration

        Yields:
            str: Token chunks as they are generated
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass
