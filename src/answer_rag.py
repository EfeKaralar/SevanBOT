#!/usr/bin/env python3
"""
RAG answer generation CLI.

Retrieves relevant chunks and generates factual, source-attributed answers
using Claude. Supports single queries, batch processing, streaming, and
side-by-side strategy comparison.

Usage:
    # Single query with hybrid retrieval (default)
    python3 src/answer_rag.py --query "Türk dili tarihi hakkında ne yazıyor?"

    # Use dense retrieval only
    python3 src/answer_rag.py --query "Osmanlı İmparatorluğu" --strategy dense

    # Compare all three strategies side-by-side
    python3 src/answer_rag.py --query "dil devrimi" --compare-strategies

    # Batch queries from a file (single strategy)
    python3 src/answer_rag.py --queries rag_queries.txt --output results/

    # Batch comparison across all strategies
    python3 src/answer_rag.py --queries rag_queries.txt --compare-strategies --output results/

    # Streaming mode
    python3 src/answer_rag.py --query "modernleşme" --stream

    # Higher quality model
    python3 src/answer_rag.py --query "test" --model claude-3-5-sonnet-20241022

    # More context chunks
    python3 src/answer_rag.py --query "dil devrimi" --top-k 20 --max-chunks 15
"""

import argparse
import json
import sys
from pathlib import Path
import dotenv

# Add project root to path and load .env
sys.path.insert(0, str(Path(__file__).parent.parent))
dotenv.load_dotenv()

from src.retrieval import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    RRFFusion,
    SearchConfig,
)
from src.rag import ClaudeAnswerGenerator, GenerationConfig


# ---------------------------------------------------------------------------
# Retriever helpers
# ---------------------------------------------------------------------------

def create_retriever(args):
    """Create retriever based on strategy argument."""
    if args.strategy == "dense":
        print("[INIT] Creating dense retriever...")
        dense = DenseRetriever(
            collection_name=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            embedding_model=args.embedding_model,
        )
        print(f"[INIT] Dense retriever ready: {dense.get_collection_info()}")
        return dense

    elif args.strategy == "sparse":
        print("[INIT] Creating sparse retriever...")
        sparse = SparseRetriever(
            chunks_file=args.chunks_file,
            use_stemming=not args.no_stemming,
        )
        print(f"[INIT] Sparse retriever ready: {sparse.get_index_stats()}")
        return sparse

    elif args.strategy == "hybrid":
        print("[INIT] Creating dense retriever...")
        dense = DenseRetriever(
            collection_name=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            embedding_model=args.embedding_model,
        )
        print(f"[INIT] Dense retriever ready: {dense.get_collection_info()}")

        print("[INIT] Creating sparse retriever...")
        sparse = SparseRetriever(
            chunks_file=args.chunks_file,
            use_stemming=not args.no_stemming,
        )
        print(f"[INIT] Sparse retriever ready: {sparse.get_index_stats()}")

        print("[INIT] Creating hybrid retriever (RRF fusion)...")
        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion_strategy=RRFFusion(),
        )
        print("[INIT] Hybrid retriever ready")
        return hybrid

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def create_all_retrievers(args) -> dict:
    """Create all three retrievers for strategy comparison."""
    print("[INIT] Creating dense retriever...")
    dense = DenseRetriever(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
    )
    print(f"[INIT] Dense retriever ready: {dense.get_collection_info()}")

    print("[INIT] Creating sparse retriever...")
    sparse = SparseRetriever(
        chunks_file=args.chunks_file,
        use_stemming=not args.no_stemming,
    )
    print(f"[INIT] Sparse retriever ready: {sparse.get_index_stats()}")

    print("[INIT] Creating hybrid retriever (RRF fusion)...")
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion_strategy=RRFFusion(),
    )
    print("[INIT] Hybrid retriever ready")

    return {"dense": dense, "sparse": sparse, "hybrid": hybrid}


def retrieval_results_to_chunks(retrieval_response) -> list:
    """Convert RetrievalResponse results to the chunk dict format for RAG."""
    chunks = []
    for result in retrieval_response.results:
        chunks.append(
            {
                "chunk_id": result.chunk_id,
                "text": result.content,
                "title": result.metadata.get("title", ""),
                "date": result.metadata.get("date", ""),
                "score": result.score,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Query runners
# ---------------------------------------------------------------------------

def run_single_query(query: str, retriever, generator, args):
    """Retrieve + generate for a single query, then display."""
    print(f"\n[QUERY] {query}")
    print(f"[RETRIEVAL] Using {args.strategy} strategy...")

    search_config = SearchConfig(
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
    )
    retrieval_response = retriever.search_with_timing(query, search_config)

    print(
        f"[RETRIEVAL] Found {len(retrieval_response.results)} chunks "
        f"in {retrieval_response.total_time_ms:.0f}ms"
    )

    if not retrieval_response.results:
        print("[WARNING] No chunks retrieved. Try a different query or strategy.")
        return

    chunks = retrieval_results_to_chunks(retrieval_response)

    gen_config = GenerationConfig(
        model=args.model,
        max_context_chunks=args.max_chunks,
        citation_format=args.citation_format,
        stream=args.stream,
    )

    print(f"[GENERATION] Using {args.model}...")

    if args.stream:
        print(f"\n{'='*80}")
        print(f"SORU: {query}")
        print(f"{'='*80}\n")
        for token in generator.generate_streaming(query, chunks, gen_config):
            print(token, end="", flush=True)
        print(f"\n{'='*80}\n")
    else:
        rag_response = generator.generate(query, chunks, gen_config)

        # Fill in retrieval timing
        rag_response.retrieval_strategy = args.strategy
        rag_response.retrieval_time_ms = retrieval_response.total_time_ms
        rag_response.total_time_ms = (
            retrieval_response.total_time_ms + rag_response.generation_time_ms
        )

        print(f"\n{'='*80}")
        print(rag_response.format_for_display())
        print(f"{'='*80}\n")

        if args.output:
            _export_response(rag_response, Path(args.output), f"answer_{abs(hash(query)) % 100000}")


def run_batch_queries(queries_file: str, retriever, generator, args):
    """Process a file of queries and optionally export all results."""
    queries = _load_queries(queries_file)
    print(f"[BATCH] Processing {len(queries)} queries...")

    search_config = SearchConfig(
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
    )
    gen_config = GenerationConfig(
        model=args.model,
        max_context_chunks=args.max_chunks,
        citation_format=args.citation_format,
    )

    results = []
    total_cost = 0.0

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")

        retrieval_response = retriever.search_with_timing(query, search_config)
        chunks = retrieval_results_to_chunks(retrieval_response)

        if not chunks:
            print("  [WARNING] No chunks retrieved, skipping.")
            continue

        rag_response = generator.generate(query, chunks, gen_config)
        rag_response.retrieval_strategy = args.strategy
        rag_response.retrieval_time_ms = retrieval_response.total_time_ms
        rag_response.total_time_ms = (
            retrieval_response.total_time_ms + rag_response.generation_time_ms
        )

        cost = rag_response.usage.calculate_cost()
        total_cost += cost

        print(
            f"  Cost: ${cost:.6f} | "
            f"Retrieval: {rag_response.retrieval_time_ms:.0f}ms | "
            f"Generation: {rag_response.generation_time_ms:.0f}ms"
        )

        results.append(rag_response.to_dict())

    print(f"\n[BATCH] Complete. {len(results)}/{len(queries)} queries answered.")
    print(f"[BATCH] Total cost: ${total_cost:.4f}")

    if args.output and results:
        output_path = Path(args.output) / "batch_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_queries": len(queries),
                    "answered": len(results),
                    "total_cost_usd": round(total_cost, 6),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[EXPORT] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Strategy comparison runners
# ---------------------------------------------------------------------------

def run_strategy_comparison(query: str, retrievers: dict, generator, args):
    """
    Generate answers for the same query using all three retrieval strategies.

    Displays results side-by-side and optionally exports a comparison JSON.
    """
    print(f"\n[COMPARE] Query: {query}")

    search_config = SearchConfig(
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
    )
    gen_config = GenerationConfig(
        model=args.model,
        max_context_chunks=args.max_chunks,
        citation_format=args.citation_format,
    )

    strategy_responses = {}

    for strategy_name, retriever in retrievers.items():
        print(f"[COMPARE] Running {strategy_name} strategy...")

        retrieval_response = retriever.search_with_timing(query, search_config)
        chunks = retrieval_results_to_chunks(retrieval_response)

        if not chunks:
            print(f"  [WARNING] No chunks for {strategy_name}, skipping.")
            continue

        rag_response = generator.generate(query, chunks, gen_config)
        rag_response.retrieval_strategy = strategy_name
        rag_response.retrieval_time_ms = retrieval_response.total_time_ms
        rag_response.total_time_ms = (
            retrieval_response.total_time_ms + rag_response.generation_time_ms
        )

        strategy_responses[strategy_name] = rag_response

        cost = rag_response.usage.calculate_cost()
        print(
            f"  {strategy_name}: {len(chunks)} chunks | "
            f"{rag_response.total_time_ms:.0f}ms | ${cost:.6f}"
        )

    # Display all answers
    print(f"\n{'='*80}")
    print(f"SORU: {query}")
    print(f"{'='*80}")

    for strategy_name, response in strategy_responses.items():
        print(f"\n--- [{strategy_name.upper()}] ---")
        print(response.answer)
        if response.sources:
            print("\nKaynaklar:")
            for s in response.sources[:3]:  # Show top 3 sources per strategy
                print(f"  {s.to_markdown()}")

    # Summary line
    print(f"\n{'='*80}")
    print("ÖZET:")
    for strategy_name, response in strategy_responses.items():
        cost = response.usage.calculate_cost()
        print(
            f"  {strategy_name:8s} | {response.total_time_ms:6.0f}ms | "
            f"${cost:.6f} | {len(response.sources)} kaynak"
        )
    print(f"{'='*80}\n")

    # Export
    if args.output:
        comparison = {
            "query": query,
            "strategies": {
                name: resp.to_dict()
                for name, resp in strategy_responses.items()
            },
            "summary": {
                name: {
                    "total_ms": resp.total_time_ms,
                    "cost_usd": resp.usage.calculate_cost(),
                    "sources_count": len(resp.sources),
                }
                for name, resp in strategy_responses.items()
            },
        }
        _export_json(
            comparison,
            Path(args.output),
            f"comparison_{abs(hash(query)) % 100000}",
        )

    return strategy_responses


def run_batch_comparison(queries_file: str, retrievers: dict, generator, args):
    """
    Run strategy comparison for every query in a file.

    Exports one summary JSON with all comparisons.
    """
    queries = _load_queries(queries_file)
    print(f"[BATCH] Comparing {len(queries)} queries across {len(retrievers)} strategies...")

    all_comparisons = []
    total_cost = 0.0

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")

        strategy_responses = run_strategy_comparison(query, retrievers, generator, args)

        for resp in strategy_responses.values():
            total_cost += resp.usage.calculate_cost()

        all_comparisons.append(
            {
                "query": query,
                "strategies": {
                    name: resp.to_dict()
                    for name, resp in strategy_responses.items()
                },
            }
        )

    print(f"\n[BATCH] Complete. {len(all_comparisons)} comparisons done.")
    print(f"[BATCH] Total cost: ${total_cost:.4f}")

    if args.output:
        _export_json(
            {
                "total_queries": len(queries),
                "strategies_compared": list(retrievers.keys()),
                "total_cost_usd": round(total_cost, 6),
                "comparisons": all_comparisons,
            },
            Path(args.output),
            "batch_comparison",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_queries(path: str) -> list:
    """Load queries from a text file (one per line, # = comment)."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def _export_response(response, output_dir: Path, filename: str):
    """Export a single RAGResponse to JSON."""
    _export_json(response.to_dict(), output_dir, filename)


def _export_json(data: dict, output_dir: Path, filename: str):
    """Write any dict to a JSON file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[EXPORT] Saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG answer generation using hybrid retrieval + Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Query input (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", type=str, help="Single question to answer")
    query_group.add_argument(
        "--queries", type=str, help="File with questions (one per line)"
    )

    # Retrieval strategy
    parser.add_argument(
        "--strategy",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid)",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Run all three strategies and compare answers side-by-side",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Chunks to retrieve (default: 10)"
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant (default: 60)")
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=50,
        help="Dense candidates before fusion (default: 50)",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=50,
        help="Sparse candidates before fusion (default: 50)",
    )

    # Generation config
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Claude model (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Max retrieved chunks to include in context (default: 10)",
    )
    parser.add_argument(
        "--citation-format",
        choices=["markdown", "numbered"],
        default="markdown",
        help="Source citation format (default: markdown)",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming token output"
    )

    # Data sources
    parser.add_argument(
        "--collection",
        type=str,
        default="sevanbot_openai-small",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant-url", type=str, default=None, help="Qdrant server URL"
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=".qdrant",
        help="Local Qdrant storage path (default: .qdrant)",
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="chunks_contextual.jsonl",
        help="Path to chunks file (default: chunks_contextual.jsonl)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--no-stemming",
        action="store_true",
        help="Disable Turkish stemming for sparse retrieval",
    )

    # Output
    parser.add_argument(
        "--output", type=str, help="Directory to export JSON results"
    )

    args = parser.parse_args()

    # Streaming is not supported with batch or comparison modes
    if args.stream and args.queries:
        print("[WARNING] --stream is not supported with --queries. Ignoring --stream.")
        args.stream = False
    if args.stream and args.compare_strategies:
        print("[WARNING] --stream is not supported with --compare-strategies. Ignoring --stream.")
        args.stream = False

    # Initialize generator (shared across all modes)
    print("[INIT] Creating answer generator...")
    generator = ClaudeAnswerGenerator()
    print(f"[INIT] Generator ready (model: {args.model})\n")

    if args.compare_strategies:
        # Create all three retrievers for comparison
        retrievers = create_all_retrievers(args)

        print(f"\n[CONFIG] top_k={args.top_k} | max_chunks={args.max_chunks} | "
              f"model={args.model} | mode=compare-strategies")

        if args.query:
            run_strategy_comparison(args.query, retrievers, generator, args)
        elif args.queries:
            run_batch_comparison(args.queries, retrievers, generator, args)
    else:
        # Single strategy mode
        retriever = create_retriever(args)

        print(f"[CONFIG] top_k={args.top_k} | max_chunks={args.max_chunks} | "
              f"model={args.model} | strategy={args.strategy}")

        if args.query:
            run_single_query(args.query, retriever, generator, args)
        elif args.queries:
            run_batch_queries(args.queries, retriever, generator, args)


if __name__ == "__main__":
    main()
