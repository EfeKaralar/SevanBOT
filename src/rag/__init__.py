"""
RAG answer generation module.

Provides LLM-based answer generation on top of the retrieval system.

Usage:
    from rag import ClaudeAnswerGenerator, GenerationConfig, RAGResponse
"""

from .config import GenerationConfig
from .response import RAGResponse, RAGUsageStats, SourceCitation
from .base import BaseAnswerGenerator
from .claude_generator import ClaudeAnswerGenerator
from .conversation import ConversationManager

__all__ = [
    "GenerationConfig",
    "RAGResponse",
    "RAGUsageStats",
    "SourceCitation",
    "BaseAnswerGenerator",
    "ClaudeAnswerGenerator",
    "ConversationManager",
]
