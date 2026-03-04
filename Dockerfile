FROM python:3.14-slim

WORKDIR /app

# Build tools for native extensions (pystemmer, grpcio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application source and static assets
COPY src/ ./src/
COPY static/ ./static/

# Bundle the chunk index for sparse (BM25) retrieval
COPY chunks_contextual.jsonl .

EXPOSE 8000

CMD ["python3", "src/api.py"]
