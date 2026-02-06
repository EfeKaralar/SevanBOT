# SevanBot

RAG system for querying Sevan Nisanyan's writings. Scrapes articles from Substack and sevannisanyan.com, converts them to Markdown, and provides semantic search.

## Architecture

```
Sources (Substack sitemap / SevanNisanyan.com API)
    |
    v
HTML Downloads --> Markdown Conversion --> Chunking --> pgvector (embeddings)
                                              |
                                              v
                                       REST API (FastAPI)
```

## Tech Stack

| Component | Choice | Status |
|-----------|--------|--------|
| Scraping | Python (requests, BeautifulSoup) | Done |
| Markdown conversion | Python (markdownify) | Done |
| Vector database | pgvector | Decided |
| Chunking | Semantic/recursive | Done |
| Embedding model | multilingual-e5-large-instruct | Decided |
| Query interface | FastAPI | Decided |
| Contextual retrieval | - | Planned |
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

## Notes

- pgvector chosen for simplicity: single Postgres instance handles vectors + metadata
- Using `intfloat/multilingual-e5-large-instruct` (560M params, 1024 dimensions)

