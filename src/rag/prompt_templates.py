"""
Turkish prompt templates for RAG answer generation.

Design philosophy:
- Turkish language throughout (essays are in Turkish)
- First-person AI replica persona for Sevan Nisanyan voice
- Anti-hallucination: only use provided sources
- Source-aware answers without dry citation-first tone
"""

from typing import List, Dict, Any, Optional


def build_system_prompt(
    use_sources: bool = True,
    humor_mode: bool = False,
    persona_mode: str = "impersonation",
) -> str:
    """Build a system prompt with persona and style controls."""
    if persona_mode == "assistant":
        role_block = (
            "Sen Sevan Nisanyan'ın yazılarını derinlemesine bilen bir asistansın. "
            "Türkçe, net ve doğrudan yanıt ver."
        )
    else:
        role_block = (
            "Bu sohbette Sevan Nisanyan'ın bir yapay zeka replikası olarak konuşuyorsun. "
            "Her zaman birinci tekil şahısla (ben) yanıt ver ve Sevan'ın üslubunu taklit et."
        )

    source_block = (
        "YALNIZCA verilen kaynak metinlere dayan. Kaynak dışı bilgi uydurma. "
        "Kaynaklarda çelişki varsa bunu açıkça söyle."
        if use_sources
        else
        "Kaynak metin yoksa yalnızca sohbet bağlamına dayan. "
        "Yeterli bilgi yoksa açıkça bunu söyle."
    )

    humor_block = (
        "Kullanıcı mizahi ton bekliyor; konuya bağlı, kısa ve zeki espri yapabilirsin. "
        "Mizah içeriğin özünü gölgelememeli."
        if humor_mode
        else
        "Kullanıcı mizah istemedikçe nötr ve ciddi kal."
    )

    return f"""{role_block}

Görev:
- Kullanıcı sorusunu doğrudan yanıtla.
- {source_block}
- Cevabı sorunun karmaşıklığına göre uyarlayıp basit soruda kısa, karmaşık soruda daha derinlikli yaz.
- Cevap boyunca yazı dili akıcı, kişisel ve kendinden emin olsun; kuru rapor dili kullanma.
- Gereksiz giriş cümlesi kurma, konuya doğrudan gir.
- Gerekirse kısa örneklerle aç.
- {humor_block}

Kimlik/temsil kuralı:
- Kullanıcı kimliğini sorgularsa veya "gerçekten Sevan mısın / AI mısın / onu temsil ediyor musun" benzeri bir soru sorarsa
  açıkça şunu belirt: Sevan'ın AI replikası olduğunu, hata yapabileceğini ve gerçek kişiyi birebir temsil etmediğini.
- Böyle bir soru yoksa bu açıklamayı kendiliğinden ekleme."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Build the formatted context block from retrieved chunks.

    This is the cacheable portion of the prompt - same chunks, same cache hit.

    Args:
        chunks: List of chunk dicts with keys: text, title, date, chunk_id, score

    Returns:
        Formatted context string wrapped in <kaynaklar> tags
    """
    parts = ["<kaynaklar>"]

    for i, chunk in enumerate(chunks, 1):
        parts.append(f"\n--- Kaynak {i} ---")

        if chunk.get("title"):
            parts.append(f"Makale: {chunk['title']}")
        if chunk.get("date"):
            parts.append(f"Tarih: {chunk['date']}")

        content = chunk.get("text", chunk.get("content", ""))
        parts.append(f"\n{content}")

    parts.append("\n</kaynaklar>")
    return "\n".join(parts)


def build_conversation_block(
    summary: Optional[str],
    recent_messages: Optional[List[Dict[str, str]]],
) -> str:
    parts = []
    if summary:
        parts.append("<sohbet_ozeti>")
        parts.append(summary)
        parts.append("</sohbet_ozeti>")

    if recent_messages:
        parts.append("<son_mesajlar>")
        for m in recent_messages:
            role = "Kullanıcı" if m.get("role") == "user" else "Asistan"
            parts.append(f"{role}: {m.get('content', '')}")
        parts.append("</son_mesajlar>")

    return "\n".join(parts)


def build_messages(
    query: str,
    chunks: List[Dict[str, Any]],
    use_caching: bool = True,
    conversation_summary: Optional[str] = None,
    recent_messages: Optional[List[Dict[str, str]]] = None,
    humor_mode: bool = False,
    response_instruction: Optional[str] = None,
) -> List[Dict]:
    """
    Build the messages list for the Claude API call.

    When use_caching=True, the context block is marked for prompt caching,
    making follow-up queries with the same context ~10x cheaper.

    Args:
        query: User's question
        chunks: Retrieved chunks (already limited to max_context_chunks)
        use_caching: Whether to enable prompt caching on the context block
        conversation_summary: Optional short Turkish conversation summary
        recent_messages: Optional list of recent turns
        humor_mode: Whether user intent suggests humorous tone
        response_instruction: Optional answer-style/length hint

    Returns:
        Messages list for anthropic.messages.create()
    """
    context_text = build_context_block(chunks) if chunks else ""
    conversation_text = build_conversation_block(conversation_summary, recent_messages)
    tone_text = "Yanıt tonu: mizahi ve zeki" if humor_mode else "Yanıt tonu: nötr"
    query_parts = [tone_text, f"Soru: {query}"]
    if response_instruction:
        query_parts.append(f"Yanıt yönergesi: {response_instruction}")
    query_text = "\n".join(query_parts)

    if use_caching and context_text:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context_text,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "\n\n".join([t for t in [conversation_text, query_text] if t]),
                    },
                ],
            }
        ]

    combined = "\n\n".join([t for t in [context_text, conversation_text, query_text] if t])
    return [{"role": "user", "content": combined}]
