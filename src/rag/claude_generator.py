"""
Claude-based RAG answer generator.

Uses Anthropic Claude with prompt caching and exponential backoff retry logic.
Patterns reused from contextual_utils.py.
"""

import os
import time
from typing import List, Dict, Any, Iterator, Optional

import anthropic
from anthropic import RateLimitError, APIError

from .base import BaseAnswerGenerator
from .config import GenerationConfig
from .response import RAGResponse, RAGUsageStats, SourceCitation
from .prompt_templates import SYSTEM_PROMPT_TR, build_messages


class ClaudeAnswerGenerator(BaseAnswerGenerator):
    """
    RAG answer generation using Anthropic Claude.

    Features:
    - Prompt caching on context (10x cost reduction for follow-up queries)
    - Exponential backoff retry logic (matches contextual_utils.py pattern)
    - Token usage and cost tracking
    - Streaming support
    - Turkish-optimized prompts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ):
        """
        Initialize Claude generator.

        Args:
            api_key: Anthropic API key. Reads ANTHROPIC_API_KEY from env if None.
            max_retries: Maximum retry attempts for rate limits / API errors.
            initial_retry_delay: Initial delay (seconds) for exponential backoff.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY in environment or .env file."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return "claude"

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> RAGResponse:
        """
        Generate an answer synchronously.

        Args:
            query: User's question
            chunks: Retrieved chunks (will be limited to config.max_context_chunks)
            config: Generation configuration

        Returns:
            RAGResponse with answer, sources, timing, and usage stats
        """
        start = time.time()

        chunks_to_use = chunks[: config.max_context_chunks]

        # Build messages with or without prompt caching
        messages = build_messages(
            query=query,
            chunks=chunks_to_use,
            use_caching=config.use_prompt_caching,
        )

        # Call Claude with retry logic
        response, usage = self._call_with_retry(messages, config)

        generation_time_ms = (time.time() - start) * 1000
        answer = response.content[0].text

        sources = self._build_citations(chunks_to_use)

        return RAGResponse(
            answer=answer,
            query=query,
            sources=sources,
            config=config,
            retrieval_strategy="",  # Filled in by the caller (answer_rag.py)
            model=config.model,
            generation_time_ms=generation_time_ms,
            total_time_ms=generation_time_ms,  # Retrieval time added by caller
            usage=usage,
        )

    def generate_streaming(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        """
        Generate an answer with token streaming.

        Yields token chunks as they arrive from the API. The caller is
        responsible for printing them.

        Args:
            query: User's question
            chunks: Retrieved chunks
            config: Generation configuration

        Yields:
            str: Token text chunks
        """
        chunks_to_use = chunks[: config.max_context_chunks]

        messages = build_messages(
            query=query,
            chunks=chunks_to_use,
            use_caching=config.use_prompt_caching,
        )

        extra_headers = (
            {"anthropic-beta": "prompt-caching-2024-07-31"}
            if config.use_prompt_caching
            else {}
        )

        with self.client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=SYSTEM_PROMPT_TR,
            messages=messages,
            extra_headers=extra_headers,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        messages: List[Dict],
        config: GenerationConfig,
    ):
        """
        Call the Claude API with exponential backoff retry.

        Mirrors the retry pattern from contextual_utils.py.

        Returns:
            Tuple of (anthropic.Message, RAGUsageStats)
        """
        extra_headers = (
            {"anthropic-beta": "prompt-caching-2024-07-31"}
            if config.use_prompt_caching
            else {}
        )

        delay = self.initial_retry_delay
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    system=SYSTEM_PROMPT_TR,
                    messages=messages,
                    extra_headers=extra_headers,
                )

                usage = RAGUsageStats(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_creation_tokens=getattr(
                        response.usage, "cache_creation_input_tokens", 0
                    ),
                    cache_read_tokens=getattr(
                        response.usage, "cache_read_input_tokens", 0
                    ),
                )

                return response, usage

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    print(
                        f"  [RATE_LIMIT] Attempt {attempt + 1}/{self.max_retries}, "
                        f"waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded after {self.max_retries} attempts"
                    ) from e

            except APIError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    print(
                        f"  [API_ERROR] Attempt {attempt + 1}/{self.max_retries}: {e}, "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise RuntimeError(
                        f"API error after {self.max_retries} attempts: {e}"
                    ) from e

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts: {last_exception}"
        )

    def _build_citations(self, chunks: List[Dict[str, Any]]) -> List[SourceCitation]:
        """Build SourceCitation objects from retrieved chunks."""
        citations = []
        for chunk in chunks:
            content = chunk.get("text", chunk.get("content", ""))
            citations.append(
                SourceCitation(
                    chunk_id=chunk.get("chunk_id", ""),
                    title=chunk.get("title", "Başlıksız"),
                    date=chunk.get("date"),
                    relevance_score=chunk.get("score", 0.0),
                    excerpt=content[:200],
                )
            )
        return citations
