# Conversational RAG Changes (Turkish-First)

## Summary
This repository now supports conversation-aware RAG. Follow-up questions are rewritten into standalone Turkish queries, retrieval is gated when it is unnecessary, and recent context is summarized to keep answers coherent without bloating prompts.

## Problems Addressed
- Follow-up questions lacked context and were treated as fresh searches.
- Every message triggered retrieval, even when a short clarification would suffice.
- Turkish context was not explicitly carried into retrieval or generation.

## High-Level Design
1. **Conversation Memory** keeps a short Turkish summary and a cached set of retrieved chunks.
2. **Query Rewriting** turns a follow-up into a standalone Turkish search query.
3. **Retrieval Gate** decides whether to run retrieval or reuse cached context.
4. **No-Sources Mode** answers from conversation context when retrieval is skipped.

## Key Implementation Pieces
- `src/rag/conversation.py`
  - `ConversationManager` handles summary updates, query rewriting, gating, and cached chunks.
  - Turkish cue phrases guide the retrieval gate.
- `src/rag/prompt_templates.py`
  - Adds `<sohbet_ozeti>` and `<son_mesajlar>` blocks.
  - Adds a no-sources system prompt for context-only answers.
- `src/rag/claude_generator.py`
  - Accepts conversation context and supports no-sources generation.
- `src/api.py`
  - Uses the conversation manager to decide when to retrieve.
  - Reuses cached chunks on follow-ups to reduce latency and cost.
  - Updates conversation summaries after assistant messages.

## Retrieval Gate (Turkish Heuristics)
- **Force retrieval** when queries mention sources, dates, or specific items:
  - e.g., `kaynak`, `alıntı`, `hangi yazı`, `tarih`, `makale`
- **Skip retrieval** when a short follow-up likely refers to prior context:
  - e.g., `bunu açar mısın?`, `peki ya o?`, `devam et`, `biraz daha`
- If cached chunks exist and the query is short, reuse them instead of re-retrieving.

## Query Rewriting (Turkish)
A Turkish prompt rewrites the last user message into a standalone search query using:
- Conversation summary
- Last few messages

If context is insufficient, the original query is returned unchanged.

## Conversation Summary
A compact Turkish summary is maintained to support:
- Better query rewriting
- Context-only answers when retrieval is skipped
- Stable behavior in multi-turn chat

## Cached Retrieval
When retrieval runs, the top chunks (truncated) are cached in the conversation:
- Used for follow-up questions without another retrieval round
- Keeps citations aligned with the last relevant context

## No-Sources Mode
If retrieval is skipped and no cached chunks exist:
- The generator answers from conversation context only
- It must say it lacks sufficient info when needed

## Turkish-First Considerations
- All prompts and summaries are in Turkish.
- Gate heuristics include Turkish follow-up language.
- Rewriter outputs Turkish standalone queries.

## Files Changed
- `src/rag/conversation.py` (new)
- `src/rag/prompt_templates.py` (conversation blocks + no-sources prompt)
- `src/rag/claude_generator.py` (conversation-aware generation)
- `src/api.py` (retrieval gate + caching + summary updates)
- `src/rag/__init__.py` (export `ConversationManager`)

## Expected Impact
- **Higher quality follow-ups**: queries preserve context.
- **Lower latency and cost**: retrieval skipped when unnecessary.
- **Improved Turkish coherence**: summaries and prompts are Turkish-first.

## Next Ideas
- Replace heuristics with a small classifier for gating.
- Add a per-conversation retrieval cache TTL.
- Add evaluation scripts for multi-turn accuracy.
