# SevanBot

RAG-powered chat app for querying Sevan Nisanyan's writings (~1,300 articles, ~1.1M words). Scrapes articles from Substack and sevannisanyan.com, converts them to Markdown, indexes them with hybrid search, and serves a streaming chat interface at `http://localhost:8000`.

## Architecture

```
Sources (Substack sitemap / SevanNisanyan.com API)
    |
    v
HTML Downloads --> Markdown Conversion --> Chunking --> Qdrant (vector DB)
                                              |              |
                                      OpenAI Embeddings   BM25 Index
                                                               |
                                                        FastAPI + SSE
                                                               |
                                                      Chat UI (browser)
```

## Tech Stack

| Component | Choice | Status |
|-----------|--------|--------|
| Scraping | Python (requests, BeautifulSoup) | Done |
| Markdown conversion | Python (markdownify) | Done |
| Vector database | Qdrant (local or managed) | Done |
| Chunking | Semantic/recursive with overlap | Done |
| Contextual retrieval | LLM-based (Claude Haiku) | Done |
| Embedding model | OpenAI text-embedding-3-small/large | Done |
| Hybrid search | BM25 + vector (RRF fusion) | Done |
| Query interface | FastAPI + SSE streaming | Done |
| Adaptive retrieval | LLM-based query planner | Done |
| Impersonation mode | First-person responses as Sevan | Done |
| Chat web app | Single-file frontend, conversation history | Done |
| Reranking | Cross-encoder reranker | Planned |
| Evaluation harness | Recall@K, MRR metrics | Planned |

## Dataset

- ~1,300 articles
- ~1.1 million words
- Sources: Substack newsletter, sevannisanyan.com blog
- Language: Turkish

## Development Setup

```bash
# Create the environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download and convert articles (Substack)
python3 src/main.py

# Download from sevannisanyan.com
python3 src/main.py --source sevan

# Chunk documents for embedding
python3 src/chunk_documents.py                      # Simple context (default)
python3 src/chunk_documents.py --context-mode llm   # LLM-generated context

# Generate embeddings
export OPENAI_API_KEY=sk-...
python3 src/embed_documents.py --model openai-small

# Start the dev server
python3 src/api.py
# → http://localhost:8000
```

## Deployment (Docker Compose)

The recommended way to run SevanBot on a local machine or VPS is with Docker Compose.
It starts two services: the app and a Qdrant vector database.

### 1. Prerequisites

- Docker and Docker Compose installed
- `chunks_contextual.jsonl` built locally (see Development Setup above)
- Embeddings uploaded to the Qdrant instance (see step 3)

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# Required
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
APP_PASSWORD=your_password_here

# Optional
ANTHROPIC_MODEL=claude-sonnet-4-20250514   # default model for chat
# COLLECTION_NAME=sevanbot_openai-small    # override collection name
# CHUNKS_FILE=chunks_contextual.jsonl      # override chunks file path
```

### 3. Start the services

```bash
docker compose up -d
```

This starts:
- **app** — FastAPI server on port `8000` (not exposed publicly; put a reverse proxy in front)
- **qdrant** — Vector DB on `127.0.0.1:6333` (localhost only)

### 4. Load embeddings (one-time)

The Qdrant volume starts empty. Run the embedding job once to populate it:

```bash
# From the host, pointing at the local Qdrant container
QDRANT_URL=http://localhost:6333 \
OPENAI_API_KEY=sk-... \
python3 src/embed_documents.py --model openai-small
```

Or run it inside the container if you prefer:

```bash
docker compose exec app python3 src/embed_documents.py --model openai-small
```

After this, the `qdrant_data` volume persists the index across restarts.

### 5. Reverse proxy (VPS)

Expose port `8000` via nginx or Caddy with HTTPS. Example Caddy config:

```
yourdomain.com {
    reverse_proxy localhost:8000
}
```

### Incremental updates

When you add new articles and re-chunk:

```bash
python3 src/chunk_documents.py --context-mode llm
docker compose exec app python3 src/embed_documents.py --model openai-small --incremental
```

### Volumes

| Volume | Contents |
|--------|----------|
| `qdrant_data` | Qdrant index (vectors + BM25) |
| `conversations` | Chat history (one JSON per conversation) |

Back these up before re-deploying if you want to preserve data.

## Project Structure

```
src/
  api.py               # FastAPI backend (streaming chat, conversation CRUD)
  main.py              # Pipeline orchestrator
  download_articles.py # Fetches articles from Substack / SevanNisanyan.com
  convert_to_md.py     # HTML → Markdown conversion
  chunk_documents.py   # Semantic chunking with contextual enrichment
  embed_documents.py   # Embedding generation (OpenAI or local models)
  contextual_utils.py  # Claude Haiku context generation with prompt caching
  answer_rag.py        # CLI for single-query RAG testing
  test_retrieval.py    # Retrieval strategy comparison CLI
  smoke_impersonation.py # Smoke tests for chat/RAG quality
  rag/                 # Answer generation module (config, prompts, Claude client)
  retrieval/           # Retrieval module (dense, sparse, hybrid, RRF fusion)
static/
  index.html           # Single-file chat frontend (HTML + CSS + JS)
chunks_contextual.jsonl  # Chunked + enriched documents (BM25 + embedding source)
```

## Chunking Strategy

- **Primary split**: Paragraphs (semantic boundaries)
- **Fallback**: Sentences if paragraph > 400 tokens
- **Minimum size**: 150 tokens (merges small paragraphs)
- **Overlap**: 1 sentence between chunks for context continuity
- **Filtering**: Removes footnotes, images, Ottoman transliteration examples
- **Context enrichment**: Metadata prepend (simple) or LLM-generated summary (optional)

Output: `chunks_contextual.jsonl` with ~6,850 chunks (avg ~1,265 chars each)

## Planned

- **Cross-encoder reranking** — Improve answer quality by reranking retrieved chunks before generation (`bge-reranker-v2-m3` or Cohere Rerank 3.5)
- **Retrieval evaluation harness** — Automated Recall@K and MRR measurement against a hand-curated query dataset (`eval_queries.json`)
- **Streaming in impersonation mode** — Token-by-token output for the impersonation persona
- **Query decomposition** — Break complex multi-part questions into sub-queries before retrieval

## Implementation Notes

### Embedding Model

**Chosen**: OpenAI `text-embedding-3-small` (1536 dims) for production

- Zero local RAM (API-based, critical for small VPS)
- $0.05 for full corpus (~6,850 chunks); incremental updates cost <$0.001 per 100 chunks
- Multilingual: strong on Turkish via MIRACL benchmarks
- `text-embedding-3-large` (3072 dims) available for higher quality at 7x cost

### Vector Database

**Chosen**: Qdrant with named Docker volume

- Native hybrid search (dense + sparse/BM25)
- Self-hosted, no cloud dependency
- Healthcheck ensures app waits for Qdrant to be ready before starting

### Retrieval

Three strategies, selectable at query time:

| Strategy | Method | Best For |
|----------|--------|----------|
| Dense | Vector similarity (Qdrant) | Semantic / conceptual queries |
| Sparse | BM25 with Turkish stemming | Exact terms, proper nouns |
| Hybrid | RRF fusion of dense + sparse | General purpose (default) |

Adaptive retrieval uses a Claude LLM planner to choose strategy and parameters per query.
