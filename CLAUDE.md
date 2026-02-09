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
python3 src/chunk_documents.py                              # Simple context (default)
python3 src/chunk_documents.py --context-mode llm           # LLM-generated context (requires ANTHROPIC_API_KEY)
python3 src/chunk_documents.py --context-mode llm --max-docs 10  # Test on 10 docs

# Resume/checkpoint features (interrupt-safe)
python3 src/chunk_documents.py --context-mode llm           # Automatically resumes from chunks_contextual.jsonl
# Ctrl+C to stop, run again to continue - already-processed docs are skipped

# Compare context modes side-by-side
python3 src/compare_contexts.py --sample 5                  # Compare 5 random documents
python3 src/compare_contexts.py --source sevan --sample 3   # Compare 3 Sevan articles

# Install dependencies
uv pip install -r requirements.txt
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
- `src/chunk_documents.py` - Semantic chunking with paragraph/sentence splitting, overlap, noise filtering, contextual enrichment
- `src/contextual_utils.py` - LLM-based context generation using Claude Haiku with prompt caching
- `src/compare_contexts.py` - Tool to compare simple vs LLM context modes

---

## Contextual Retrieval (IMPLEMENTED ✓)

The chunking pipeline now supports two context enrichment modes:

### Simple Mode (Default)
- Prepends document metadata to each chunk: `Makale: [title] | Tarih: [date] | Konular: [keywords]`
- Instant, free, effective for most use cases
- Output: `chunks.jsonl`

### LLM Mode (Optional)
- Uses Claude Haiku to generate semantic context for each chunk
- Prompt caching reduces cost by ~10x after first chunk per document
- Turkish-language prompts optimized for essay content
- Cost: ~$2-3 for full corpus (~6,850 chunks)
- Output: `chunks_contextual.jsonl`
- **Resume/checkpoint support**: Interrupt-safe, skips already-processed documents

**Setup for LLM mode:**
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run with LLM context (can interrupt with Ctrl+C and resume)
python3 src/chunk_documents.py --context-mode llm

# Progress is saved incrementally - safe to stop/resume anytime
```

**Resume functionality:**
- Automatically detects existing `chunks_contextual.jsonl`
- Skips documents that already have chunks in the file
- No re-processing or duplicate API calls
- Perfect for slow/unreliable networks

**Compare modes:**
```bash
python3 src/compare_contexts.py --sample 5
```

Shows side-by-side comparison of simple vs LLM-generated context with cost estimates.

---

---

## Embedding Generation (IMPLEMENTED ✓)

The embedding pipeline supports **local models** (GPU/CPU) and **API-based models** (OpenAI) with **incremental updates**.

### Available Models

| Model | Type | Dimensions | RAM Usage | Cost | Speed |
|-------|------|------------|-----------|------|-------|
| `turkembed` | Local | 768 | ~2GB | Free | Medium |
| `bge-m3` | Local | 1024 | ~3GB | Free | Medium |
| `openai-small` | API | 1536 | 0MB | $0.05/full corpus | Fast |
| `openai-large` | API | 3072 | 0MB | $0.35/full corpus | Fast |

### Usage

```bash
# OpenAI Embeddings (recommended for low RAM systems)
export OPENAI_API_KEY=sk-your-key-here
python3 src/embed_documents.py --model openai-small
python3 src/embed_documents.py --model openai-large

# Local Models (free, but requires RAM)
python3 src/embed_documents.py --model turkembed
python3 src/embed_documents.py --model bge-m3

# Incremental embedding (only embed new chunks)
python3 src/embed_documents.py --model openai-small --incremental

# Force re-embed everything
python3 src/embed_documents.py --model openai-small --force

# Test with small file
python3 src/embed_documents.py --model openai-small --chunks-file test_chunks.jsonl --skip-qdrant
```

### Cost Analysis (Full Corpus: 6,850 chunks)

- **Initial embedding**: $0.05 (openai-small) or $0.35 (openai-large)
- **Incremental updates** (100 new chunks): $0.0008 (small) or $0.005 (large)
- **Re-embedding everything**: Same as initial (avoid with `--incremental` flag)

### Incremental Updates

The pipeline automatically tracks which chunks are already embedded:

```bash
# First run: Embed all 6,850 chunks
python3 src/embed_documents.py --model openai-small

# Later: Add 100 new articles, re-chunk
python3 src/chunk_documents.py --context-mode llm

# Only embed the new chunks (saves money!)
python3 src/embed_documents.py --model openai-small --incremental
```

Output is saved to `embeddings/{model}/embeddings.jsonl` and metadata to `embeddings/{model}/metadata.json`.

---

## RAG Retrieval System (IMPLEMENTED ✓)

A modular, production-ready retrieval system with dense, sparse, and hybrid search capabilities.

### Architecture

```
INDEXING: Documents → Semantic Chunking → Context Enrichment → Dual Index
                           ↓                      ↓                    ↓
                    256-512 tokens         LLM/Simple Context   Qdrant + BM25

RETRIEVAL: Query → Hybrid Search → RRF Fusion → Top-K Results
                        ↓              ↓
                 Dense + Sparse    Combine scores
```

### Quick Start

```bash
# Dense retrieval (vector similarity)
python3 src/test_retrieval.py --query "Türk dili tarihi" --strategy dense

# Sparse retrieval (BM25 keyword matching)
python3 src/test_retrieval.py --query "Osmanlı İmparatorluğu" --strategy sparse

# Hybrid retrieval (combines dense + sparse with RRF)
python3 src/test_retrieval.py --query "modernleşme" --strategy hybrid

# Compare all strategies side-by-side
python3 src/test_retrieval.py --query "dil devrimi" --compare

# Batch comparison with example queries
python3 src/test_retrieval.py --queries example_queries.txt --compare --output results/
```

### Available Retrieval Strategies

| Strategy | Method | Best For | Latency |
|----------|--------|----------|---------|
| **Dense** | Vector similarity (Qdrant) | Semantic search, conceptual queries | ~50-100ms |
| **Sparse** | BM25 keyword matching (Turkish stemming) | Exact terms, proper nouns | ~20-50ms |
| **Hybrid** | RRF fusion of dense + sparse | General purpose, best recall | ~100-150ms |

### Configuration Options

```bash
# Adjust number of results
--top-k 20

# Change RRF constant (default: 60)
--rrf-k 30

# Adjust retrieval before fusion (default: 50 each)
--dense-top-k 100 --sparse-top-k 100

# Use weighted fusion instead of RRF (experimental)
--strategy hybrid --fusion weighted --dense-weight 0.7 --sparse-weight 0.3

# Export results to JSON for analysis
--output results/
```

### Modular Design

The system is built with clean abstractions for experimentation:

```
src/retrieval/
├── base.py          # Abstract classes (BaseRetriever, BaseFusion)
├── dense.py         # DenseRetriever (Qdrant vector search)
├── sparse.py        # SparseRetriever (BM25 with Turkish stemming)
├── fusion.py        # RRFFusion, WeightedFusion
├── hybrid.py        # HybridRetriever (combines retrievers)
└── evaluator.py     # Comparison and metrics tools
```

**Programmatic Usage:**

```python
from retrieval import DenseRetriever, SparseRetriever, HybridRetriever, SearchConfig

# Initialize
dense = DenseRetriever(collection_name="contextual_chunks")
sparse = SparseRetriever(chunks_file="chunks_contextual.jsonl")
hybrid = HybridRetriever(dense, sparse)

# Search
config = SearchConfig(top_k=10, rrf_k=60)
results = hybrid.search("Türk dili tarihi", config)

# Compare
from retrieval import RetrievalComparator
comparator = RetrievalComparator({"dense": dense, "sparse": sparse, "hybrid": hybrid})
comparison = comparator.compare("query", config)
comparison.print_summary()
```

### Key Technical Decisions

| Component | Course Approach (Outdated) | Modern Approach (Use This) |
|-----------|---------------------------|---------------------------|
| Embedding | `text-embedding-ada-002`, `all-MiniLM-L6-v2` | `text-embedding-3-large` or `Cohere embed-v4` |
| Hybrid Fusion | Weighted average | **Reciprocal Rank Fusion (RRF)** |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `bge-reranker-v2-m3` or `Cohere Rerank 3.5` |
| Context | Simple metadata prepending | **Both implemented:** Simple (default) or LLM-generated (optional) |

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
- ColBERT (overkill unless scaling to millions of docs)
- Query decomposition (start simple, add later if needed)
