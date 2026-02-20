"""
Adaptive response-length policy for conversational RAG.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


COMPLEXITY_CUES_TR = (
    "neden",
    "nasıl",
    "detay",
    "detaylı",
    "analiz",
    "karşılaştır",
    "bağlam",
    "tarihsel",
    "siyasi",
    "dilbilim",
    "örnek",
    "yorumla",
    "açıkla",
    "değerlendir",
)

BRIEF_CUES_TR = (
    "kısa",
    "özet",
    "tek cümle",
    "madde madde",
    "kısaca",
    "özetle",
)


@dataclass
class ResponseSettings:
    profile: str
    max_tokens: int
    response_instruction: str


def derive_response_settings(
    query: str,
    chunks: List[Dict[str, Any]],
    recent_messages: Optional[List[Dict[str, str]]] = None,
) -> ResponseSettings:
    """
    Pick token budget + style instruction based on query complexity.
    """
    text = (query or "").strip().lower()
    word_count = len(text.split())
    complexity = 0

    if word_count >= 45:
        complexity += 3
    elif word_count >= 25:
        complexity += 2
    elif word_count >= 12:
        complexity += 1

    cue_hits = sum(1 for cue in COMPLEXITY_CUES_TR if cue in text)
    complexity += min(cue_hits, 3)

    if text.count("?") >= 2:
        complexity += 1

    if len(chunks) >= 8:
        complexity += 2
    elif len(chunks) >= 4:
        complexity += 1

    if recent_messages and any(token in text for token in ("bunu", "buna", "şunu", "onu")):
        complexity += 1

    if any(cue in text for cue in BRIEF_CUES_TR):
        complexity -= 3

    if complexity <= 1:
        return ResponseSettings(
            profile="short",
            max_tokens=550,
            response_instruction=(
                "Soru basitse kısa ve net yanıt ver. Gereksiz uzatma yapma. "
                "Uygunsa bir iki cümlelik hafif, doğal bir nüans ekleyebilirsin."
            ),
        )

    if complexity <= 4:
        return ResponseSettings(
            profile="medium",
            max_tokens=1100,
            response_instruction=(
                "Orta detayda, düzenli paragraflarla yanıt ver. "
                "İddiaları kaynaklarla ilişkilendir. Ton doğal ve canlı olsun."
            ),
        )

    return ResponseSettings(
        profile="long",
        max_tokens=1900,
        response_instruction=(
            "Karmaşık soruya derinlikli yanıt ver: bağlam, karşılaştırma ve gerekçeleri açıkla. "
            "Gerektiğinde bölümleyerek yaz. Dil doğal, yer yer zekice ve ölçülü olsun."
        ),
    )
