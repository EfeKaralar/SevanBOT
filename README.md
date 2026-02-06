# SevanBot

RAG system for querying Sevan Nisanyan's writings. Scrapes articles from Substack and sevannisanyan.com, converts them to Markdown, and provides semantic search.

## Architecture

```
Sources (Substack sitemap / SevanNisanyan.com API)
    |
    v
HTML Downloads --> Markdown Conversion --> pgvector (embeddings)
    |
    v
Query Interface (TBD)
```

## Tech Stack

| Component | Choice | Status |
|-----------|--------|--------|
| Scraping | Python (requests, BeautifulSoup) | Done |
| Markdown conversion | Python (markdownify) | Done |
| Vector database | pgvector | Decided |
| Chunking | Semantic/recursive | Decided |
| Embedding model | TBD | - |
| Query interface | TBD | - |

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
```

## Project Structure

```
src/
  main.py              # Pipeline orchestrator
  download_articles.py # Fetches articles from sources
  convert_to_md.py     # HTML to Markdown conversion
sources/               # Raw HTML files
formatted/             # Converted Markdown files
  substack/
  sevan/
```

## Notes

- pgvector chosen for simplicity: single Postgres instance handles vectors + metadata
- Multilingual embedding model will be used (Turkish content, potential English queries)
- Semantic/recursive chunking: split by paragraphs, then sentences if >400 tokens. Uniform strategy for both sources yields better retrieval precision than document-as-chunk.
- Estimated ~5,000 vectors after chunking