# SevanBot

RAG system for querying Sevan Nisanyan's writings. Scrapes articles from Substack and sevannisanyan.com, converts them to Markdown, and provides semantic search.

## Architecture

```
Sources (Substack sitemap / SevanNisanyan.com API)
    |
    v
HTML Downloads --> Markdown Conversion --> Chunking --> Qdrant (vector DB)
                                              |            |
                                              v            v
                                    OpenAI Embeddings  REST API (FastAPI)
```

## Tech Stack

| Component | Choice | Status |
|-----------|--------|--------|
| Scraping | Python (requests, BeautifulSoup) | Done |
| Markdown conversion | Python (markdownify) | Done |
| Vector database | Qdrant (local storage) | Done |
| Chunking | Semantic/recursive | Done |
| Contextual retrieval | LLM-based (Claude Haiku) | Done |
| Embedding model | OpenAI text-embedding-3-small/large | Done |
| Query interface | FastAPI | Planned |
| Hybrid search | BM25 + vector | Planned |
| Reranking | - | Planned |

## Dataset

- ~1,300 articles
- ~1.1 million words
- Sources: Substack newsletter, sevannisanyan.com blog
- Language: Turkish

## Usage
Uvicorn is the recommended package manager for this project. You can use any other of your choosing

```bash
# Create the environment
uv venv

# Set the environment
source ./venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download and convert articles (Substack)
python3 src/main.py

# Download from sevannisanyan.com
python3 src/main.py --source sevan

# Download only / Convert only
python3 src/main.py --skip-convert
python3 src/main.py --skip-download

# Delete HTML after conversion
python3 src/main.py --delete-after

# Chunk documents for embedding
python3 src/chunk_documents.py
```

## Project Structure

```
src/
  main.py              # Pipeline orchestrator
  download_articles.py # Fetches articles from sources
  convert_to_md.py     # HTML to Markdown conversion
  chunk_documents.py   # Semantic chunking for embeddings
sources/               # Raw HTML files
formatted/             # Converted Markdown files
  substack/
  sevan/
chunks.jsonl           # Chunked documents ready for embedding
```

## Chunking Strategy

Documents are chunked for optimal RAG performance:

- **Primary split**: Paragraphs (semantic boundaries)
- **Fallback**: Sentences if paragraph > 400 tokens
- **Minimum size**: 150 tokens (merges small paragraphs)
- **Overlap**: 1 sentence between chunks for context continuity
- **Filtering**: Removes footnotes, images, Ottoman transliteration examples

Output: `chunks.jsonl` with ~7,100 chunks (avg 1,265 chars each)

## TODO: Retrieval Evaluation (Post-Deployment)

Once the RAG system is operational, implement quantitative evaluation to measure retrieval quality:

### Evaluation Metrics
- **Recall@K**: % of relevant chunks retrieved in top K results (target: >90% @ K=10)
- **MRR (Mean Reciprocal Rank)**: Average position of first relevant result (target: >0.7)
- **NDCG**: Ranking quality metric
- **Latency**: End-to-end retrieval time (target: <500ms)

### Evaluation Dataset
Create ground truth dataset with 50-100 real user queries:
```json
{
  "query": "Osmanlı'da azınlık hakları",
  "relevant_docs": ["doc_123", "doc_456"],
  "difficulty": "medium"
}
```

### Comparison Scenarios
1. **Simple vs LLM context**: Does LLM-generated context improve retrieval accuracy?
2. **Dense vs Hybrid search**: Does BM25 + vector fusion outperform vector-only?
3. **With vs without reranking**: Does cross-encoder reranking improve top-K results?

### Implementation
- `src/evaluate_retrieval.py`: Automated evaluation harness
- `eval_queries.json`: Hand-curated query dataset
- Track metrics over time to measure improvements

**Data collection strategy**: Log real user queries + clicked results to build ground truth organically.

---

## Implementation Notes

### Embedding Model Decision

**Chosen**: OpenAI `text-embedding-3-small` (1536 dims) / `text-embedding-3-large` (3072 dims)

**Rationale**:
- **Performance**: text-embedding-3-large ranks 2nd globally on MTEB (64.6%), outperforming multilingual-e5-large-instruct (62%)
- **Multilingual**: Massive improvements (31.4% → 54.9% on MIRACL benchmark)
- **Operational**: Zero local RAM usage (critical for 8GB systems), API-based
- **Cost-effective**: $0.05 (small) / $0.35 (large) for full corpus (~6,850 chunks)
- **Incremental updates**: Only pay for new chunks (<$0.001 for 100 new chunks)

- **text-embedding-3-small (1536 dims)**
  - Cost: $0.05 for full corpus
  - Speed: Faster inference
  - Lower Qdrant storage
  - Quality: 62.3% MTEB score

- **text-embedding-3-large (3072 dims)**
  - Cost: $0.35 for full corpus (7x more)
  - Storage: 2x dimensions = 2x Qdrant disk space
  - Quality: 64.6% MTEB score (+2.3% better retrieval)
  - Turkish: Likely better on multilingual tasks

**Alternative considered**: `intfloat/multilingual-e5-large-instruct` (560M params, 1024 dims) - leads Turkish TR-MTEB benchmarks but requires 2-3GB RAM and doesn't support incremental updates without re-embedding.

### Vector Database Decision

**Chosen**: Qdrant with local storage

**Rationale**:
- Native hybrid search support (dense + sparse/BM25)
- Local deployment option (no cloud dependency)
- Better suited for RAG: optimized for vector similarity + filtering
- pgvector alternative considered but Qdrant offers better retrieval performance

### Contextual Retrieval

Implemented LLM-based context generation using Claude Haiku:
- Generates semantic context for each chunk
- Prompt caching reduces cost by ~10x
- Cost: ~$2-3 for full corpus
- Fallback: Simple metadata prepending (free, instant)

