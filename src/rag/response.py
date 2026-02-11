"""
Response dataclasses for RAG answer generation.

Mirrors the RetrievalResponse pattern from retrieval/base.py.
"""

from dataclasses import dataclass, field
from typing import ClassVar, List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .config import GenerationConfig


@dataclass
class SourceCitation:
    """Represents a cited source chunk in the RAG response."""

    chunk_id: str
    title: str
    date: Optional[str]
    relevance_score: float
    excerpt: str  # First 200 chars of chunk content

    def to_markdown(self) -> str:
        """Format as markdown citation: **Title** (date) (score: 0.874)"""
        date_str = f" ({self.date})" if self.date else ""
        return f"- **{self.title}**{date_str} (score: {self.relevance_score:.3f})"

    def to_numbered(self, index: int) -> str:
        """Format as numbered citation: [1] Title"""
        date_str = f" ({self.date})" if self.date else ""
        return f"[{index}] {self.title}{date_str}"


@dataclass
class RAGUsageStats:
    """
    Token usage and cost tracking for a single RAG generation call.

    Pricing is model-aware: pass the model ID used for generation so that
    costs are computed with the correct per-token rates.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = "claude-3-5-haiku-20241022"

    # Pricing per 1M tokens keyed by model ID (input, output, cache_write, cache_read)
    _MODEL_PRICING: ClassVar[Dict[str, Dict[str, float]]] = {
        "claude-3-haiku-20240307":     {"input": 0.25,  "output": 1.25,  "cache_write": 0.30,  "cache_read": 0.03},
        "claude-3-5-haiku-20241022":   {"input": 0.80,  "output": 4.00,  "cache_write": 1.00,  "cache_read": 0.08},
        "claude-3-5-sonnet-20241022":  {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
        "claude-3-opus-20240229":      {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    }
    # Fallback for unknown models (haiku rates)
    _DEFAULT_PRICING: ClassVar[Dict[str, float]] = {
        "input": 0.80, "output": 4.00, "cache_write": 1.00, "cache_read": 0.08
    }

    # Set in __post_init__ based on model
    INPUT_PRICE: float = field(default=0.0, init=False, repr=False)
    OUTPUT_PRICE: float = field(default=0.0, init=False, repr=False)
    CACHE_WRITE_PRICE: float = field(default=0.0, init=False, repr=False)
    CACHE_READ_PRICE: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        pricing = self._MODEL_PRICING.get(self.model, self._DEFAULT_PRICING)
        self.INPUT_PRICE = pricing["input"]
        self.OUTPUT_PRICE = pricing["output"]
        self.CACHE_WRITE_PRICE = pricing["cache_write"]
        self.CACHE_READ_PRICE = pricing["cache_read"]

    def calculate_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.input_tokens / 1_000_000) * self.INPUT_PRICE
        output_cost = (self.output_tokens / 1_000_000) * self.OUTPUT_PRICE
        cache_write_cost = (self.cache_creation_tokens / 1_000_000) * self.CACHE_WRITE_PRICE
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * self.CACHE_READ_PRICE
        return input_cost + output_cost + cache_write_cost + cache_read_cost

    def to_dict(self) -> Dict[str, Any]:
        """Export for JSON serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": round(self.calculate_cost(), 6),
        }


@dataclass
class RAGResponse:
    """
    Complete response from RAG answer generation.

    Includes the generated answer, source citations, performance metrics,
    and cost tracking.
    """

    # Core response
    answer: str
    query: str
    sources: List[SourceCitation]

    # Configuration used
    config: "GenerationConfig"
    retrieval_strategy: str  # "dense" | "sparse" | "hybrid"
    model: str

    # Performance metrics (ms)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Cost tracking
    usage: RAGUsageStats = field(default_factory=RAGUsageStats)

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def format_for_display(self) -> str:
        """Format answer with citations and metrics for CLI display."""
        lines = []
        lines.append(f"SORU: {self.query}")
        lines.append("")
        lines.append("CEVAP:")
        lines.append(self.answer)

        if self.sources:
            lines.append("")
            lines.append("KAYNAKLAR:")
            for i, source in enumerate(self.sources, 1):
                if self.config.citation_format == "numbered":
                    lines.append(source.to_numbered(i))
                else:
                    lines.append(source.to_markdown())

        lines.append("")
        lines.append("---")
        cost = self.usage.calculate_cost()
        lines.append(
            f"*Süre: {self.total_time_ms:.0f}ms | "
            f"Maliyet: ${cost:.6f} | "
            f"Model: {self.model}*"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "title": s.title,
                    "date": s.date,
                    "score": s.relevance_score,
                    "excerpt": s.excerpt,
                }
                for s in self.sources
            ],
            "performance": {
                "retrieval_ms": self.retrieval_time_ms,
                "generation_ms": self.generation_time_ms,
                "total_ms": self.total_time_ms,
            },
            "usage": self.usage.to_dict(),
            "config": {
                "model": self.model,
                "retrieval_strategy": self.retrieval_strategy,
                "temperature": self.config.temperature,
                "max_context_chunks": self.config.max_context_chunks,
            },
            "timestamp": self.timestamp,
        }
