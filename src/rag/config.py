"""
Configuration dataclasses for RAG answer generation.

Mirrors the SearchConfig pattern from retrieval/base.py.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """
    Configuration for RAG answer generation.

    Controls model selection, context size, and output formatting.
    """

    # Model selection
    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.0  # Deterministic for factual answers
    max_tokens: int = 2048

    # Context configuration
    max_context_chunks: int = 10  # How many retrieved chunks to include
    include_metadata: bool = True  # Include title/date in context

    # Response formatting
    citation_format: str = "markdown"  # "markdown" | "numbered"

    # Streaming
    stream: bool = False

    # Cost control
    use_prompt_caching: bool = True  # Cache context (10x cheaper for follow-ups)
