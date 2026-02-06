# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Article scraper, converter, and RAG pipeline. Downloads articles from Substack (via sitemap) or SevanNisanyan.com (via API), converts HTML to clean Markdown, and provides retrieval-augmented generation capabilities.

## Commands

```bash
# Full pipeline - Substack (default)
python3 src/main.py                        # Download batch of 10, convert

# SevanNisanyan.com source
python3 src/main.py --source sevan         # Download from sevannisanyan.com
python3 src/main.py --source sevan --keywords "Pazar Sohbeti"  # With filter

# Custom batch size
python3 src/main.py --batch-size 20

# Delete HTML files after conversion (saves storage)
python3 src/main.py --delete-after

# Download only / Convert only
python3 src/main.py --skip-convert
python3 src/main.py --skip-download

# Chunk documents for embedding
python3 src/chunk_documents.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Scraping Pipeline:**
- Substack: `sitemap.xml` → `download_articles.py` → HTML → `convert_to_md.py` → Markdown
- SevanNisanyan.com: `API (__data.json)` → `download_articles.py` → HTML → `convert_to_md.py` → Markdown
- Chunking: Markdown → `chunk_documents.py` → `chunks.jsonl`

**Files:**
- `src/main.py` - Orchestrates batch loop: download N files → convert → delete (optional) → repeat until done
- `src/download_articles.py` - Downloads articles from Substack (sitemap) or SevanNisanyan.com (API)
- `src/convert_to_md.py` - Extracts title/subtitle/date/content, converts to Markdown
- `src/chunk_documents.py` - Semantic chunking with paragraph/sentence splitting, overlap, noise filtering

---

## RAG Retrieval Pipeline (To Implement)

See `advanced-rag/RETRIEVAL_PLAN.md` for full details. This section summarizes key decisions for implementation.

### Target Architecture

```
INDEXING: Documents → Semantic Chunking → Context Enrichment → Dual Index (Dense + Sparse)
                           ↓                      ↓                    ↓
                    256-512 tokens         Prepend metadata      Vector + BM25

RETRIEVAL: Query → Hybrid Search → RRF Fusion → Reranking → Top-K Results
                        ↓              ↓            ↓
                 Dense + Sparse    Combine      Cross-encoder
```

### Implementation Phases

**Phase 1: Foundation**
1. Set up vector database (Qdrant recommended - has native hybrid support)
2. Choose embedding model: `text-embedding-3-large` (OpenAI) or `Cohere embed-v4` (excellent multilingual)
3. Implement basic dense retrieval from existing chunks

**Phase 2: Hybrid Search**
4. Add BM25 sparse indexing using `bm25s` library
5. Implement Reciprocal Rank Fusion (RRF) - NOT weighted averaging
6. Test hybrid vs dense-only

**Phase 3: Reranking**
7. Add cross-encoder reranking: `BAAI/bge-reranker-v2-m3` (open-source) or `Cohere Rerank 3.5` (API)
8. Retrieve 20-50, rerank to top 5

**Phase 4: Advanced (Optional)**
9. Query expansion with LLM
10. Contextual embeddings experiment

### Key Technical Decisions

| Component | Course Approach (Outdated) | Modern Approach (Use This) |
|-----------|---------------------------|---------------------------|
| Embedding | `text-embedding-ada-002`, `all-MiniLM-L6-v2` | `text-embedding-3-large` or `Cohere embed-v4` |
| Hybrid Fusion | Weighted average | **Reciprocal Rank Fusion (RRF)** |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `bge-reranker-v2-m3` or `Cohere Rerank 3.5` |
| Context | LLM-generated per chunk (expensive) | **Prepend document metadata** (simple, effective) |

### RRF Implementation (Critical)

```python
def reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """Combine ranked lists using RRF. k=60 is research-backed default."""
    fused_scores = {}
    for results in results_lists:
        for rank, result in enumerate(results, 1):
            doc_id = result['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'doc': result, 'score': 0}
            fused_scores[doc_id]['score'] += 1 / (rank + k)
    return sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
```

### Context Enrichment (Simple Approach)

```python
def create_chunk_with_context(chunk: str, doc_metadata: dict) -> str:
    """Prepend document-level context to chunk before embedding."""
    parts = []
    if doc_metadata.get("title"):
        parts.append(f"Article: {doc_metadata['title']}")
    if doc_metadata.get("date"):
        parts.append(f"Date: {doc_metadata['date']}")
    if doc_metadata.get("subtitle"):
        parts.append(f"Summary: {doc_metadata['subtitle']}")
    return f"{' | '.join(parts)}\n\n{chunk}"
```

### Turkish Language Considerations

- Use multilingual embedding models (OpenAI, Cohere, or `BAAI/bge-m3`)
- Test chunk boundaries with actual Turkish articles
- BM25 may need Turkish stemmer (PyStemmer supports Turkish)
- Skip Reverse HyDE - essays/opinions don't benefit from hypothetical questions

### Libraries to Use

```
qdrant-client    # Vector database with native hybrid search
bm25s            # Fast BM25 implementation
sentence-transformers  # For cross-encoder reranking
openai           # For embeddings (text-embedding-3-large)
# OR
cohere           # Alternative: embed-v4 + Rerank 3.5
```

### Metrics to Track

| Metric | Target |
|--------|--------|
| Recall@10 | >90% |
| MRR | >0.7 |
| Latency | <500ms |

### What NOT to Implement

- Reverse HyDE (not suited for essay content)
- LLM-generated context per chunk (too expensive for this use case)
- ColBERT (overkill unless scaling to millions of docs)
- Query decomposition (start simple)
