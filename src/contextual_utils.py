#!/usr/bin/env python3
"""
Contextual retrieval utilities for SevanBot.

Provides LLM-based context generation for chunks using Anthropic Claude Haiku
with prompt caching to reduce costs.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import anthropic


# Turkish prompt templates for context generation
DOCUMENT_CONTEXT_PROMPT_TR = """
<belge>
{doc_content}
</belge>
"""

CHUNK_CONTEXT_PROMPT_TR = """
İşte belgenin tamamı içinde konumlandırmak istediğimiz parça:
<parca>
{chunk_content}
</parca>

Lütfen bu parçayı belgenin tamamı içinde konumlandıran kısa ve öz bir bağlam cümlesi verin.
Amaç, bu parçanın arama sonuçlarında daha iyi bulunmasını sağlamaktır.
Sadece bağlam cümlesini verin, başka bir şey eklemeyin.
"""


@dataclass
class UsageStats:
    """Token usage statistics for a single API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: 'UsageStats') -> None:
        """Add another UsageStats to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens


class ContextGenerationStats:
    """Track token usage and costs for LLM-based context generation."""

    # Pricing per 1M tokens (as of January 2025 for Claude Haiku)
    INPUT_PRICE = 0.25  # $0.25 per 1M tokens
    OUTPUT_PRICE = 1.25  # $1.25 per 1M tokens
    CACHE_WRITE_PRICE = 0.30  # $0.30 per 1M tokens
    CACHE_READ_PRICE = 0.03  # $0.03 per 1M tokens (10x cheaper!)

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.num_requests = 0

    def add_usage(self, usage: UsageStats) -> None:
        """Add usage stats from an API call."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_creation_tokens += usage.cache_creation_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.num_requests += 1

    def calculate_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.INPUT_PRICE
        output_cost = (self.total_output_tokens / 1_000_000) * self.OUTPUT_PRICE
        cache_write_cost = (self.total_cache_creation_tokens / 1_000_000) * self.CACHE_WRITE_PRICE
        cache_read_cost = (self.total_cache_read_tokens / 1_000_000) * self.CACHE_READ_PRICE

        return input_cost + output_cost + cache_write_cost + cache_read_cost

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0-1)."""
        total_cacheable = self.total_cache_creation_tokens + self.total_cache_read_tokens
        if total_cacheable == 0:
            return 0.0
        return self.total_cache_read_tokens / total_cacheable

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "requests": self.num_requests,
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cache_creation": self.total_cache_creation_tokens,
                "cache_read": self.total_cache_read_tokens,
            },
            "cost_usd": round(self.calculate_cost(), 4),
            "cache_hit_rate": round(self.cache_hit_rate(), 3),
        }


def situate_context(
    doc_content: str,
    chunk_text: str,
    title: str,
    anthropic_client: anthropic.Anthropic
) -> Tuple[str, UsageStats]:
    """
    Generate contextual description for a chunk using Claude Haiku.

    Uses prompt caching to cache the full document content, making subsequent
    chunks from the same document ~10x cheaper.

    Args:
        doc_content: Full document text (will be cached)
        chunk_text: The specific chunk to contextualize
        title: Document title (for logging/debugging)
        anthropic_client: Anthropic API client

    Returns:
        Tuple of (context_text, usage_stats)

    Raises:
        anthropic.APIError: If the API call fails
    """
    try:
        response = anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT_TR.format(doc_content=doc_content),
                            "cache_control": {"type": "ephemeral"}  # Cache the full document
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT_TR.format(chunk_content=chunk_text),
                        }
                    ]
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Extract context text from response
        context_text = response.content[0].text

        # Extract usage stats
        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0),
        )

        return context_text, usage

    except anthropic.APIError as e:
        # Re-raise with more context
        raise RuntimeError(f"Failed to generate context for document '{title}': {e}") from e


def get_anthropic_client() -> Optional[anthropic.Anthropic]:
    """
    Get Anthropic API client from environment.

    Returns:
        Anthropic client if API key is set, None otherwise
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def validate_context(context: str, chunk_text: str) -> bool:
    """
    Validate that generated context is reasonable.

    Args:
        context: Generated context text
        chunk_text: Original chunk text

    Returns:
        True if context passes validation, False otherwise
    """
    # Context should not be empty
    if not context or not context.strip():
        return False

    # Context should not be too long (max 500 chars is reasonable for Turkish)
    if len(context) > 500:
        return False

    # Context should not just be a copy of the chunk
    if context == chunk_text or context in chunk_text:
        return False

    # Context should be shorter than the chunk
    if len(context) >= len(chunk_text):
        return False

    return True
