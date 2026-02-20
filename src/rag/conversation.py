"""
Conversation-aware helpers for RAG.

Provides:
- Lightweight conversation memory (summary + cached retrieval)
- Query rewriting into standalone Turkish questions
- Retrieval gating for follow-up turns
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import anthropic
from anthropic import APIError, RateLimitError


@dataclass
class ConversationMemory:
    summary: str = ""
    last_retrieval_query: str = ""
    last_retrieval_chunks: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "last_retrieval_query": self.last_retrieval_query,
            "last_retrieval_chunks": self.last_retrieval_chunks or [],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConversationMemory":
        if not data:
            return cls(summary="", last_retrieval_query="", last_retrieval_chunks=[])
        return cls(
            summary=data.get("summary", ""),
            last_retrieval_query=data.get("last_retrieval_query", ""),
            last_retrieval_chunks=data.get("last_retrieval_chunks", []) or [],
        )


class ConversationManager:
    """
    Maintains conversation memory and enables conversational RAG behavior.

    Uses Claude for query rewriting + summary updates if an API key is available.
    Falls back to deterministic heuristics otherwise.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_retries: int = 2,
        initial_retry_delay: float = 0.5,
    ):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def ensure_memory(self, conv: Dict[str, Any]) -> ConversationMemory:
        memory = ConversationMemory.from_dict(conv.get("memory"))
        conv["memory"] = memory.to_dict()
        return memory

    def get_recent_messages(self, conv: Dict[str, Any], limit: int = 6) -> List[Dict[str, str]]:
        messages = conv.get("messages", [])
        trimmed = [m for m in messages if m.get("role") in ("user", "assistant")]
        return [{"role": m["role"], "content": m.get("content", "")} for m in trimmed[-limit:]]

    def cache_retrieval(
        self,
        conv: Dict[str, Any],
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10,
        max_chars: int = 1200,
    ) -> None:
        limited = []
        for c in chunks[:max_chunks]:
            text = c.get("text", c.get("content", ""))
            limited.append({
                "chunk_id": c.get("chunk_id", ""),
                "text": text[:max_chars],
                "title": c.get("title", ""),
                "date": c.get("date", ""),
                "score": c.get("score", 0.0),
            })
        conv["memory"] = ConversationMemory(
            summary=conv.get("memory", {}).get("summary", ""),
            last_retrieval_query=query,
            last_retrieval_chunks=limited,
        ).to_dict()

    def get_cached_chunks(self, conv: Dict[str, Any]) -> List[Dict[str, Any]]:
        memory = ConversationMemory.from_dict(conv.get("memory"))
        return memory.last_retrieval_chunks or []

    # ------------------------------------------------------------------
    # Conversational logic
    # ------------------------------------------------------------------

    def rewrite_query(
        self,
        message: str,
        summary: str,
        recent_messages: List[Dict[str, str]],
    ) -> str:
        if not self.client:
            return message

        system_prompt = (
            "Sen Türkçe bir sorgu yeniden yazım asistanısın. "
            "Görevin, kullanıcının son mesajını, sohbet bağlamını kullanarak "
            "tek başına anlaşılır, açık bir arama sorgusuna dönüştürmektir. "
            "Bağlam yetersizse orijinal soruyu aynen döndür. "
            "SADECE yeniden yazılmış sorguyu döndür."
        )

        convo_lines = []
        if summary:
            convo_lines.append(f"Sohbet özeti: {summary}")
        if recent_messages:
            convo_lines.append("Son mesajlar:")
            for m in recent_messages:
                role = "Kullanıcı" if m["role"] == "user" else "Asistan"
                convo_lines.append(f"- {role}: {m['content']}")

        user_prompt = "\n".join(convo_lines + [f"Kullanıcı sorusu: {message}"])

        try:
            response = self._call_model(system_prompt, user_prompt)
            rewritten = response.strip()
            return rewritten or message
        except Exception:
            return message

    def update_summary(
        self,
        previous_summary: str,
        recent_messages: List[Dict[str, str]],
    ) -> str:
        if not recent_messages:
            return previous_summary

        if not self.client:
            # Simple deterministic fallback
            tail = " ".join([m["content"] for m in recent_messages[-4:]])
            combined = (previous_summary + " " + tail).strip()
            return combined[:700]

        system_prompt = (
            "Sen Türkçe bir sohbet özeti asistanısın. "
            "Önceki özeti ve son mesajları kullanarak güncel, kısa bir özet üret. "
            "Özet 600 karakteri geçmesin. "
            "Sadece özeti döndür."
        )

        convo_lines = []
        if previous_summary:
            convo_lines.append(f"Önceki özet: {previous_summary}")
        convo_lines.append("Son mesajlar:")
        for m in recent_messages:
            role = "Kullanıcı" if m["role"] == "user" else "Asistan"
            convo_lines.append(f"- {role}: {m['content']}")

        user_prompt = "\n".join(convo_lines)

        try:
            response = self._call_model(system_prompt, user_prompt)
            summary = response.strip()
            return summary or previous_summary
        except Exception:
            return previous_summary

    # ------------------------------------------------------------------
    # Internal model call
    # ------------------------------------------------------------------

    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
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

        raise RuntimeError(f"Model call failed: {last_exception}")
