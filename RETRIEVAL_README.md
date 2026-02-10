# Retrieval System Documentation

A production-ready, modular retrieval system for semantic search over Turkish essay corpus with **5,323 contextually-enriched chunks**.

## Overview

This system implements state-of-the-art hybrid search combining:
- **Dense retrieval**: Vector similarity using Qdrant (semantic understanding)
- **Sparse retrieval**: BM25 keyword matching with Turkish stemming (exact terms)
- **Hybrid fusion**: Reciprocal Rank Fusion (RRF) for optimal recall

## Quick Start

### Prerequisites

1. **Qdrant running** (for dense search):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Dependencies installed**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Environment variables**:
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```

### Basic Usage

```bash
# Sparse retrieval (works without Qdrant, fast)
python3 src/test_retrieval.py --query "Türk dili tarihi" --strategy sparse

# Dense retrieval (requires Qdrant running)
python3 src/test_retrieval.py --query "Osmanlı İmparatorluğu" --strategy dense

# Hybrid retrieval (best results)
python3 src/test_retrieval.py --query "modernleşme" --strategy hybrid

# Compare all strategies
python3 src/test_retrieval.py --query "dil devrimi" --compare
```

### Batch Comparison

```bash
# Run multiple queries and save results
python3 src/test_retrieval.py --queries example_queries.txt --compare --output results/
```

## Architecture

### Modular Design

```
src/retrieval/
├── base.py          # Abstract base classes and data models
│   ├── BaseRetriever        # Interface for all retrievers
│   ├── BaseFusion           # Interface for fusion strategies
│   ├── SearchConfig         # Configuration dataclass
│   ├── RetrievalResult      # Standardized result format
│   └── RetrievalResponse    # Complete response with timing
│
├── dense.py         # Dense vector search via Qdrant
│   └── DenseRetriever       # Embeds query, searches Qdrant
│
├── sparse.py        # BM25 keyword search
│   └── SparseRetriever      # Turkish stemming + BM25 index
│
├── fusion.py        # Result fusion strategies
│   ├── RRFFusion            # Reciprocal Rank Fusion (default)
│   └── WeightedFusion       # Weighted score averaging
│
├── hybrid.py        # Combines dense + sparse
│   └── HybridRetriever      # Parallel search + fusion
│
└── evaluator.py     # Comparison and metrics
    ├── RetrievalComparator  # Side-by-side comparison
    └── ComparisonResult     # Analysis and export
```

### Key Design Principles

1. **Single Responsibility**: Each retriever does one thing well
2. **Open/Closed**: Easy to add new strategies without modifying existing code
3. **Dependency Injection**: Pass dependencies rather than hardcoding
4. **Common Interface**: All retrievers implement `BaseRetriever`
5. **Standardized Results**: `RetrievalResult` enables fair comparison

## Configuration Options

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 10 | Number of final results to return |
| `--min-score` | 0.0 | Minimum score threshold |
| `--dense-top-k` | 50 | Results from dense before fusion |
| `--sparse-top-k` | 50 | Results from sparse before fusion |
| `--rrf-k` | 60 | RRF constant (research-backed) |
| `--dense-weight` | 0.5 | Weight for dense (weighted fusion only) |
| `--sparse-weight` | 0.5 | Weight for sparse (weighted fusion only) |

### Data Sources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--collection` | contextual_chunks | Qdrant collection name |
| `--qdrant-url` | http://localhost:6333 | Qdrant server URL |
| `--chunks-file` | chunks_contextual.jsonl | Chunks file for BM25 |
| `--embedding-model` | text-embedding-3-small | OpenAI embedding model |
| `--no-stemming` | False | Disable Turkish stemming |

## Programmatic Usage

### Simple Search

```python
from retrieval import DenseRetriever, SparseRetriever, SearchConfig

# Dense search
dense = DenseRetriever(collection_name="contextual_chunks")
config = SearchConfig(top_k=10)
results = dense.search("Türk dili tarihi", config)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Title: {result.metadata['title']}")
    print(f"Content: {result.content[:200]}...")
```

### Hybrid Search

```python
from retrieval import DenseRetriever, SparseRetriever, HybridRetriever, RRFFusion

# Initialize retrievers
dense = DenseRetriever()
sparse = SparseRetriever(chunks_file="chunks_contextual.jsonl")

# Create hybrid with RRF fusion
hybrid = HybridRetriever(dense, sparse, fusion_strategy=RRFFusion())

# Search
config = SearchConfig(top_k=10, rrf_k=60)
results = hybrid.search("modernleşme", config)
```

### Compare Strategies

```python
from retrieval import RetrievalComparator

# Set up retrievers
retrievers = {
    "dense": DenseRetriever(),
    "sparse": SparseRetriever(chunks_file="chunks_contextual.jsonl"),
    "hybrid": HybridRetriever(dense, sparse)
}

# Compare
comparator = RetrievalComparator(retrievers)
comparison = comparator.compare("dil devrimi", config)

# View results
comparison.print_summary()

# Analyze overlap
overlap = comparison.analyze_overlap()
print(f"Jaccard similarity: {overlap['pairwise_overlap']['dense_vs_sparse']['jaccard']:.3f}")

# Export for analysis
comparison.export_to_json("results/comparison.json")
```

### Batch Evaluation

```python
queries = [
    "Türk dili tarihi",
    "Osmanlı İmparatorluğu",
    "modernleşme",
    "dil devrimi"
]

results = comparator.batch_compare(
    queries,
    config=config,
    output_dir="results/"
)

# Print aggregate stats
comparator.print_summary_stats(results)
```

## Performance

### Latency Benchmarks (MacBook Air M1)

| Strategy | Avg Latency | Components |
|----------|-------------|------------|
| Sparse | ~30ms | BM25 search |
| Dense | ~80ms | Query embedding (40ms) + Qdrant search (40ms) |
| Hybrid | ~120ms | Dense (80ms) + Sparse (30ms) + Fusion (10ms) |

### Accuracy Characteristics

| Strategy | Best For | Strengths | Weaknesses |
|----------|----------|-----------|------------|
| Dense | Semantic queries, concepts | Understands meaning, handles synonyms | Can miss exact terms |
| Sparse | Exact terms, proper nouns | Fast, precise keyword matching | No semantic understanding |
| Hybrid | General purpose | Best recall, combines both approaches | Slightly slower |

## How Hybrid Search Works

### Reciprocal Rank Fusion (RRF)

RRF is a research-backed method that combines rankings robustly:

```python
def reciprocal_rank_fusion(results_lists, k=60):
    """
    Combine multiple ranked lists using RRF.

    For each document, sum the reciprocal of its rank in each list:
        score(doc) = Σ 1/(k + rank_i)

    k=60 is the research-backed default constant.
    """
    fused_scores = {}
    for results in results_lists:
        for rank, result in enumerate(results, 1):
            doc_id = result.chunk_id
            fused_scores[doc_id] = fused_scores.get(doc_id, 0)
            fused_scores[doc_id] += 1 / (k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

**Why RRF?**
- No score normalization needed (unlike weighted fusion)
- Robust to score scale differences between retrievers
- Single parameter (k) with good default
- Research shows it outperforms weighted averaging

**Reference**: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet and individual systems"

### Example: RRF in Action

```
Query: "Türk dili tarihi"

Dense results:              Sparse results:
1. doc_42 (0.89)           1. doc_15 (12.3)
2. doc_15 (0.85)           2. doc_42 (11.8)
3. doc_7  (0.82)           3. doc_102 (10.5)

RRF fusion (k=60):
doc_42: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
doc_15: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
doc_7:  1/(60+3) + 0        = 0.0159
doc_102: 0 + 1/(60+3)       = 0.0159

Final ranking: doc_42, doc_15, doc_7, doc_102
```

Documents appearing in both lists get boosted regardless of original score magnitudes.

## Turkish Language Support

### BM25 with Turkish Stemming

The sparse retriever uses **PyStemmer** for Turkish stemming:

```python
# Example: Turkish stemming in action
"dillerinden" → "dil"
"kitapları" → "kitap"
"geliyordu" → "gel"
```

This improves recall by matching morphological variants:
- Query: "dil tarihi" also matches "dillerin tarihinde", "dillerinin tarihleri"

### Multilingual Embeddings

Dense retrieval uses `text-embedding-3-small` which has strong Turkish support:
- Trained on multilingual corpus
- Understands Turkish semantics
- Handles code-switching (Turkish/Ottoman Turkish/loan words)

## Result Analysis

### Understanding Result Objects

```python
@dataclass
class RetrievalResult:
    chunk_id: str          # Unique chunk identifier
    score: float           # Final ranking score
    content: str           # Chunk text content
    metadata: dict         # Title, date, keywords, etc.

    # Provenance tracking (for analysis)
    retrieval_method: str  # "dense", "sparse", "hybrid_rrf"
    dense_score: float     # Original dense score
    sparse_score: float    # Original sparse score
    dense_rank: int        # Rank in dense results
    sparse_rank: int       # Rank in sparse results
```

### Analyzing Overlap

```python
comparison = comparator.compare("query", config)
overlap = comparison.analyze_overlap()

print(overlap)
# {
#   "total_unique_chunks": 87,
#   "pairwise_overlap": {
#     "dense_vs_sparse": {
#       "intersection": 15,        # Docs in both
#       "union": 87,               # Total unique docs
#       "jaccard": 0.172,          # Similarity coefficient
#       "only_in_first": 35,       # Dense-only docs
#       "only_in_second": 37       # Sparse-only docs
#     }
#   }
# }
```

**Interpretation**:
- Low Jaccard (< 0.3): Dense and sparse find different documents → hybrid valuable
- High Jaccard (> 0.7): Dense and sparse agree → hybrid provides less benefit

## Exporting Results

### JSON Export Format

```bash
python3 src/test_retrieval.py --query "test" --compare --output results/
```

Creates: `results/comparison.json`

```json
{
  "query": "Türk dili tarihi",
  "strategies": {
    "dense": {
      "results": [...],
      "performance": {"total_time_ms": 85.3}
    },
    "sparse": {...},
    "hybrid": {...}
  },
  "overlap_analysis": {...}
}
```

Use this for:
- Building evaluation datasets
- Computing custom metrics (MRR, NDCG, etc.)
- Analyzing failure cases
- Tuning hyperparameters

## Next Steps

### Phase 3: Cross-Encoder Reranking

Add a reranking stage to improve precision:

```python
from retrieval import RerankedRetriever

reranker = RerankedRetriever(
    base_retriever=hybrid,
    model="BAAI/bge-reranker-v2-m3",  # Turkish-capable
    retrieve_k=50,  # Get 50 candidates
    return_k=10     # Rerank to top 10
)
```

**Benefits**:
- Improves precision at top positions
- Can fix fusion errors
- Better for question answering

**Cost**:
- Adds latency (~100-200ms)
- Requires GPU for best performance

### Phase 4: Evaluation Metrics

Build proper evaluation with ground truth:

```python
from retrieval import RetrievalEvaluator

evaluator = RetrievalEvaluator(
    retrievers=retrievers,
    queries=test_queries,
    ground_truth=relevance_judgments
)

metrics = evaluator.compute_metrics()
print(f"Recall@10: {metrics['recall@10']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
```

### Phase 5: Query Expansion

Use LLM to expand queries before search:

```python
# Original query
query = "dil devrimi"

# Expanded with Claude
expanded = expand_query(query)
# "dil devrimi Türkçe latin alfabesi harf devrimi 1928 Atatürk"

# Search with expansion
results = hybrid.search(expanded, config)
```

### Optional: Turkish Stop Word Removal

Experiment with removing common Turkish stop words (`ve`, `bir`, `bu`, `için`, etc.) in sparse retrieval to reduce noise, though BM25's IDF scoring already handles this to some extent. Add as a configurable option and compare results using the existing comparison tools.

## Troubleshooting

### Qdrant Connection Refused

```bash
# Start Qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Verify it's running
curl http://localhost:6333/health
```

### BM25 Index Build Fails

Check chunks file format:
```bash
head -1 chunks_contextual.jsonl | python3 -m json.tool
```

Required fields:
- `text_for_embedding` or `text` or `content`
- `chunk_id`
- `title`, `date`, `keywords` (optional, for metadata)

### OpenAI API Errors

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test manually
python3 -c "from openai import OpenAI; print(OpenAI().models.list())"
```

## References

### Papers

- Cormack et al. (2009): "Reciprocal Rank Fusion outperforms Condorcet"
- Gao et al. (2021): "Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline"
- Thakur et al. (2021): "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"

### Libraries

- `bm25s`: Fast BM25 implementation
- `PyStemmer`: Snowball stemmer with Turkish support
- `qdrant-client`: Vector database client
- `openai`: Embedding generation

## License

Part of SevanBot project. See main README for license information.
