"""
Configuration dataclasses for RAG answer generation.

Mirrors the SearchConfig pattern from retrieval/base.py.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """
    Configuration for RAG answer generation.

    Controls model selection, context size, and output formatting.
    """

    # Model selection
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0  # Deterministic for factual answers
    max_tokens: int = 2048

    # Context configuration
    max_context_chunks: int = 10  # How many retrieved chunks to include
    include_metadata: bool = True  # Include title/date in context

    # Response formatting
    citation_format: str = "markdown"  # "markdown" | "numbered"
    response_instruction: Optional[str] = None

    # Streaming
    stream: bool = False

    # Cost control
    use_prompt_caching: bool = True  # Cache context (10x cheaper for follow-ups)

    # Persona / style controls
    persona_mode: str = "impersonation"  # "impersonation" | "assistant"
    humor_mode: bool = False  # Enable topic-relevant humor when requested
