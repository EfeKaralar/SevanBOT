"""
Turkish prompt templates for RAG answer generation.

Design philosophy:
- Turkish language throughout (essays are in Turkish)
- Specialized for Sevan Nisanyan's essay style (historical, linguistic, political)
- Anti-hallucination: only use provided sources
- Source attribution: cite specific passages when making claims
"""

from typing import List, Dict, Any


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


def build_messages(
    query: str,
    chunks: List[Dict[str, Any]],
    use_caching: bool = True
) -> List[Dict]:
    """
    Build the messages list for the Claude API call.

    When use_caching=True, the context block is marked for prompt caching,
    making follow-up queries with the same context ~10x cheaper.

    Args:
        query: User's question
        chunks: Retrieved chunks (already limited to max_context_chunks)
        use_caching: Whether to enable prompt caching on the context block

    Returns:
        Messages list for anthropic.messages.create()
    """
    context_text = build_context_block(chunks)
    query_text = f"Soru: {query}"

    if use_caching:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context_text,
                        "cache_control": {"type": "ephemeral"},  # Cache the context
                    },
                    {
                        "type": "text",
                        "text": query_text,  # Not cached - varies per query
                    },
                ],
            }
        ]
    else:
        return [
            {
                "role": "user",
                "content": f"{context_text}\n\n{query_text}",
            }
        ]
