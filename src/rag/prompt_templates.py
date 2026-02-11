"""
Turkish prompt templates for RAG answer generation.

Design philosophy:
- Turkish language throughout (essays are in Turkish)
- Specialized for Sevan Nisanyan's essay style (historical, linguistic, political)
- Anti-hallucination: only use provided sources
- Source attribution: cite specific passages when making claims
"""

from typing import List, Dict, Any, Optional


SYSTEM_PROMPT_TR = """Sen Sevan Nisanyan'ın yazılarını derinlemesine bilen uzman bir asistansın.

Görevin, kullanıcı sorularını sağlanan kaynak metinlere dayanarak yanıtlamaktır.

Kurallar:
1. YALNIZCA sağlanan kaynak metinlerde yer alan bilgileri kullan
2. Kaynaklarda olmayan bilgileri uydurma veya genel bilgilerinle yanıtlama
3. Emin olmadığın durumda "Bu konuda verilen kaynaklarda bilgi bulamadım" de
4. Cevaplarını Türkçe, net ve anlaşılır bir dille yaz
5. Kaynaklarda çelişkili bilgi varsa, her iki tarafı da belirt
6. İddialarını desteklerken kaynaklardan alıntı yap

Yanıt formatı:
- Doğrudan soruyu yanıtla (gereksiz giriş cümleleri kullanma)
- Paragraf şeklinde, okumayı kolaylaştıran bir yapıda yaz
- Gerektiğinde madde işaretleri kullan
- Önemli terimleri **kalın** yap"""


SYSTEM_PROMPT_TR_NO_SOURCES = """Sen Sevan Nisanyan'ın yazılarını bilen bir asistansın.

Görevin, kullanıcı sorularını yalnızca sohbet bağlamına dayanarak yanıtlamaktır.

Kurallar:
1. Kaynak metin verilmediyse yalnızca sohbet bağlamını kullan
2. Bilgi yoksa açıkça "Bu konuda sohbet bağlamında yeterli bilgi yok" de
3. Cevaplarını Türkçe, net ve anlaşılır bir dille yaz
4. Gerektiğinde kullanıcıdan netleştirme iste"""


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

    Returns:
        Messages list for anthropic.messages.create()
    """
    context_text = build_context_block(chunks) if chunks else ""
    conversation_text = build_conversation_block(conversation_summary, recent_messages)
    query_text = f"Soru: {query}"

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
