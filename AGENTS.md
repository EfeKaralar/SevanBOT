# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the pipeline, retrieval, and API code. Key entry points include `src/main.py` (scrape/convert pipeline), `src/chunk_documents.py` (chunking), `src/embed_documents.py` (embeddings), `src/answer_rag.py` (RAG answers), and `src/api.py` (FastAPI chat app).
- `src/retrieval/` and `src/rag/` contain modular retrieval and answer-generation components.
- `sources/` stores raw HTML; `formatted/` stores Markdown output; `embeddings/` stores embedding artifacts.
- `static/` is the single-page web UI. `conversations/` stores persisted chat sessions as JSON.

## Build, Test, and Development Commands
- `uv venv` and `source ./venv/bin/activate`: create/activate the virtualenv.
- `uv pip install -r requirements.txt`: install dependencies.
- `python3 src/main.py`: run the scrape → convert pipeline (supports flags like `--source sevan`).
- `python3 src/chunk_documents.py`: create chunk files (`chunks.jsonl` / `chunks_contextual.jsonl`).
- `python3 src/embed_documents.py --model openai-small`: generate embeddings and load into Qdrant.
- `python3 src/answer_rag.py --query "..."`: run a single RAG query from the CLI.
- `python3 src/api.py`: start the FastAPI server at `http://localhost:9000`.

## Coding Style & Naming Conventions
- Python code follows PEP 8: 4-space indentation, `snake_case` for functions/vars, `CapWords` for classes.
- Keep modules small and focused (pipeline vs retrieval vs API). Prefer explicit imports from `src/`.
- No formatter or linter is enforced; keep diffs tidy and avoid unrelated reformatting.

## Testing Guidelines
- There is no pytest suite yet. Use the retrieval harness for validation.
- `python3 src/test_retrieval.py --query "..." --compare` runs side-by-side strategy comparisons.
- When adding features, include a small CLI example in docs or comments that proves it works.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and scoped (e.g., “Add …”, “Fix …”).
- PRs should describe the change, list key commands run, and include UI screenshots when touching `static/`.
- Link relevant issues or notes when behavior or data formats change.

## Configuration & Secrets
- Use a `.env` file for keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `APP_PASSWORD`).
- Do not commit credentials; sample setup is documented in `CLAUDE.md`.
