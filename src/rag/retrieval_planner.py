"""
LLM-based retrieval planner for adaptive RAG.

Decides when and how much to retrieve based on conversation context:
- none: 0 docs (follow-ups using existing context)
- few: 5 docs (simple factual questions)
- normal: 10 docs (standard questions)
- deep: 20 docs (deep dives, comprehensive analysis)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any

import anthropic
from anthropic import APIError, RateLimitError


RetrievalTier = Literal["none", "few", "normal", "deep"]

# Tier configuration: maps tier to retrieval parameters
TIER_CONFIG: Dict[RetrievalTier, Dict[str, int]] = {
    "none": {"top_k": 0, "max_chunks": 0},
    "few": {"top_k": 5, "max_chunks": 5},
    "normal": {"top_k": 10, "max_chunks": 10},
    "deep": {"top_k": 20, "max_chunks": 15},
}


@dataclass
class RetrievalPlan:
    """Result of the retrieval planning decision."""
    tier: RetrievalTier
    rewritten_query: Optional[str]  # Contextual query for retrieval (None if tier=none)
    reasoning: str  # Explanation for debugging/logging

    @property
    def top_k(self) -> int:
        """Number of documents to retrieve."""
        return TIER_CONFIG[self.tier]["top_k"]

    @property
    def max_chunks(self) -> int:
        """Maximum chunks to include in context."""
        return TIER_CONFIG[self.tier]["max_chunks"]

    @property
    def should_retrieve(self) -> bool:
        """Whether retrieval is needed."""
        return self.tier != "none"


# Turkish prompt for the retrieval planner
PLANNER_SYSTEM_PROMPT = """Sen bir RAG (Retrieval-Augmented Generation) sisteminde arama kararları veren bir asistansın.

Görevin, kullanıcının sorusuna cevap vermek için:
- Mevcut önbellek yeterli mi?
- Yeni kaynak araması gerekli mi?
- Gerekiyorsa kaç kaynak gerekli?

## Seviyeler
- "none": Soru önbellekteki kaynaklarla veya konuşma bağlamıyla cevaplanabilir. Takip soruları, açıklama istekleri, veya mevcut bağlamla ilgili sorular.
- "few": Basit, tek konulu soru. 5 kaynak yeterli.
- "normal": Standart soru, orta düzey bağlam gerekli. 10 kaynak.
- "deep": Kullanıcı açıkça daha fazla istedi (detaylı, kapsamlı, araştır, derinlemesine gibi kelimeler) veya karmaşık, çok yönlü bir soru. 20 kaynak.

## Önemli
- Eğer önbellekte ilgili kaynaklar varsa ve soru bunlarla cevaplanabilirse "none" seç.
- Eğer kullanıcı "bunu açar mısın", "devam et", "biraz daha" gibi takip soruları soruyorsa "none" seç.
- Eğer kullanıcı yeni bir konu soruyorsa veya önbellekte ilgili kaynak yoksa arama yap.
- Eğer tier "none" ise rewritten_query null olmalı.

Cevabını SADECE JSON formatında ver, başka bir şey yazma."""


def _build_user_prompt(
    query: str,
    conversation_summary: str,
    cached_sources_summary: str,
    recent_messages: List[Dict[str, str]],
) -> str:
    """Build the user prompt for the planner."""
    parts = []

    # Conversation summary
    if conversation_summary:
        parts.append(f"## Konuşma Özeti\n{conversation_summary}")
    else:
        parts.append("## Konuşma Özeti\nYeni konuşma, özet yok.")

    # Cached sources
    if cached_sources_summary:
        parts.append(f"## Önbellekteki Kaynaklar\n{cached_sources_summary}")
    else:
        parts.append("## Önbellekteki Kaynaklar\nÖnbellekte kaynak yok.")

    # Recent messages
    if recent_messages:
        msg_lines = ["## Son Mesajlar"]
        for m in recent_messages[-4:]:  # Last 4 messages
            role = "Kullanıcı" if m.get("role") == "user" else "Asistan"
            content = m.get("content", "")[:200]  # Truncate long messages
            msg_lines.append(f"- {role}: {content}")
        parts.append("\n".join(msg_lines))

    # Current query
    parts.append(f"## Şu Anki Soru\n{query}")

    # Expected output format
    parts.append("""## Cevap Formatı
{
  "tier": "none|few|normal|deep",
  "rewritten_query": "bağımsız arama sorgusu veya null",
  "reasoning": "kısa açıklama"
}""")

    return "\n\n".join(parts)


def summarize_cached_chunks(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    """Create a brief summary of cached chunks for the planner."""
    if not chunks:
        return ""

    titles = []
    for chunk in chunks[:max_chunks]:
        title = chunk.get("title", "")
        if title and title not in titles:
            titles.append(title)

    if not titles:
        return f"{len(chunks)} kaynak önbellekte."

    return f"Önbellekte {len(chunks)} kaynak var. Makaleler: {', '.join(titles[:3])}"


class RetrievalPlanner:
    """
    LLM-based planner that decides retrieval tier for each query.

    Uses Claude Haiku for fast, cheap decision-making (~200-400ms, ~$0.0003/call).
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-3-haiku-20240307",
        max_retries: int = 2,
        initial_retry_delay: float = 0.5,
        enabled: bool = True,
    ):
        """
        Initialize the retrieval planner.

        Args:
            client: Anthropic client (created from env if not provided)
            model: Model to use for planning decisions
            max_retries: Number of retry attempts for API errors
            initial_retry_delay: Initial delay for exponential backoff
            enabled: If False, always returns "normal" tier (for rollback)
        """
        if client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        else:
            self.client = client

        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.enabled = enabled

    def plan(
        self,
        query: str,
        conversation_summary: str = "",
        cached_chunks: Optional[List[Dict[str, Any]]] = None,
        recent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> RetrievalPlan:
        """
        Decide retrieval tier for a query.

        Args:
            query: The user's current question
            conversation_summary: Summary of conversation so far
            cached_chunks: Previously retrieved chunks (if any)
            recent_messages: Recent conversation messages

        Returns:
            RetrievalPlan with tier, rewritten_query, and reasoning
        """
        # Fallback if disabled or no client
        if not self.enabled or not self.client:
            return RetrievalPlan(
                tier="normal",
                rewritten_query=query,
                reasoning="Adaptive retrieval disabled, using default tier",
            )

        # Build prompts
        cached_summary = summarize_cached_chunks(cached_chunks or [])
        user_prompt = _build_user_prompt(
            query=query,
            conversation_summary=conversation_summary,
            cached_sources_summary=cached_summary,
            recent_messages=recent_messages or [],
        )

        try:
            response_text = self._call_model(PLANNER_SYSTEM_PROMPT, user_prompt)
            return self._parse_response(response_text, query)
        except Exception as e:
            # On any error, fall back to normal retrieval
            print(f"[Planner] Error: {e}, falling back to normal tier")
            return RetrievalPlan(
                tier="normal",
                rewritten_query=query,
                reasoning=f"Fallback due to error: {str(e)[:100]}",
            )

    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude with retry logic."""
        delay = self.initial_retry_delay
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except (RateLimitError, APIError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        raise RuntimeError(f"Planner model call failed: {last_exception}")

    def _parse_response(self, response_text: str, original_query: str) -> RetrievalPlan:
        """Parse JSON response from the model."""
        try:
            # Clean up response (remove markdown code blocks if present)
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            data = json.loads(text)

            tier = data.get("tier", "normal")
            if tier not in TIER_CONFIG:
                tier = "normal"

            rewritten_query = data.get("rewritten_query")
            if tier == "none":
                rewritten_query = None
            elif not rewritten_query:
                rewritten_query = original_query

            reasoning = data.get("reasoning", "")

            return RetrievalPlan(
                tier=tier,
                rewritten_query=rewritten_query,
                reasoning=reasoning,
            )
        except json.JSONDecodeError:
            # If parsing fails, default to normal with original query
            return RetrievalPlan(
                tier="normal",
                rewritten_query=original_query,
                reasoning=f"Failed to parse response: {response_text[:100]}",
            )
