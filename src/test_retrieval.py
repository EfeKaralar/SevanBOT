#!/usr/bin/env python3
"""
Test and compare retrieval strategies.

This CLI tool allows experimentation with different retrieval configurations:
- Dense-only, sparse-only, or hybrid search
- Different fusion methods (RRF vs weighted)
- Different hyperparameters (rrf_k, weights, top_k)
- Single queries or batch comparison
- Export results for analysis

Usage:
    # Dense retrieval only
    python3 src/test_retrieval.py --query "Türk dili tarihi" --strategy dense

    # Sparse retrieval only
    python3 src/test_retrieval.py --query "Osmanlı İmparatorluğu" --strategy sparse

    # Hybrid with RRF (default)
    python3 src/test_retrieval.py --query "modernleşme" --strategy hybrid

    # Hybrid with weighted fusion
    python3 src/test_retrieval.py --query "modernleşme" --strategy hybrid --fusion weighted

    # Compare all strategies side-by-side
    python3 src/test_retrieval.py --query "dil devrimi" --compare

    # Batch comparison with multiple queries
    python3 src/test_retrieval.py --queries queries.txt --compare --output results/

    # Custom hyperparameters
    python3 src/test_retrieval.py --query "test" --strategy hybrid --rrf-k 30 --top-k 20
"""

import argparse
import json
import sys
from pathlib import Path
import dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Load environment variables from .env
dotenv.load_dotenv()

from src.retrieval import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    RRFFusion,
    WeightedFusion,
    SearchConfig,
    RetrievalComparator,
)


def create_retrievers(args):
    """Create retriever instances based on command-line arguments."""
    retrievers = {}

    # Dense retriever
    if args.strategy in ['dense', 'hybrid', 'all'] or args.compare:
        print("[INIT] Creating dense retriever...")
        dense = DenseRetriever(
            collection_name=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            embedding_model=args.embedding_model,
        )
        retrievers['dense'] = dense
        print(f"[INIT] Dense retriever ready: {dense.get_collection_info()}")

    # Sparse retriever
    if args.strategy in ['sparse', 'hybrid', 'all'] or args.compare:
        print("[INIT] Creating sparse retriever...")
        sparse = SparseRetriever(
            chunks_file=args.chunks_file,
            use_stemming=args.use_stemming,
        )
        retrievers['sparse'] = sparse
        print(f"[INIT] Sparse retriever ready: {sparse.get_index_stats()}")

    # Hybrid retriever
    if args.strategy == 'hybrid' or args.compare:
        print(f"[INIT] Creating hybrid retriever (fusion: {args.fusion})...")

        # Choose fusion strategy
        if args.fusion == 'rrf':
            fusion = RRFFusion()
        elif args.fusion == 'weighted':
            fusion = WeightedFusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {args.fusion}")

        hybrid = HybridRetriever(
            dense_retriever=retrievers['dense'],
            sparse_retriever=retrievers['sparse'],
            fusion_strategy=fusion,
        )
        retrievers['hybrid'] = hybrid
        print(f"[INIT] Hybrid retriever ready")

    return retrievers


def create_config(args):
    """Create SearchConfig from command-line arguments."""
    return SearchConfig(
        top_k=args.top_k,
        min_score=args.min_score,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rrf_k=args.rrf_k,
    )


def run_single_query(query, retrievers, config, args):
    """Run a single query with specified strategy."""
    if args.compare:
        # Compare all strategies
        comparator = RetrievalComparator(retrievers)
        result = comparator.compare(query, config)
        result.print_summary()

        # Print overlap analysis
        overlap = result.analyze_overlap()
        if overlap:
            print(f"\n{'='*80}")
            print("OVERLAP ANALYSIS")
            print(f"{'='*80}\n")
            print(f"Total unique chunks across all strategies: {overlap['total_unique_chunks']}")
            print("\nPairwise Overlap:")
            for pair, stats in overlap['pairwise_overlap'].items():
                print(f"  {pair}:")
                print(f"    Jaccard similarity: {stats['jaccard']:.3f}")
                print(f"    Intersection: {stats['intersection']}")
                print(f"    Only in first: {stats['only_in_first']}")
                print(f"    Only in second: {stats['only_in_second']}")

        # Export if requested
        if args.output:
            output_path = Path(args.output) / "comparison.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.export_to_json(output_path)

    else:
        # Single strategy
        retriever = retrievers.get(args.strategy)
        if not retriever:
            print(f"[ERROR] Strategy '{args.strategy}' not available")
            return

        print(f"\n[SEARCH] Using {args.strategy} retrieval...")
        response = retriever.search_with_timing(query, config)

        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"STRATEGY: {args.strategy}")
        print(f"{'='*80}\n")

        print(f"Time: {response.total_time_ms:.1f}ms")
        if response.dense_time_ms > 0:
            print(f"  Dense: {response.dense_time_ms:.1f}ms")
            print(f"  Sparse: {response.sparse_time_ms:.1f}ms")
            print(f"  Fusion: {response.fusion_time_ms:.1f}ms")
        print(f"Results: {len(response.results)}\n")

        # Show results
        for i, result in enumerate(response.results, 1):
            print(f"{i}. [Score: {result.score:.4f}]")

            if result.metadata:
                title = result.metadata.get('title', 'No title')
                date = result.metadata.get('date', 'No date')
                print(f"   Title: {title}")
                print(f"   Date: {date}")

            # Show provenance for hybrid
            if result.retrieval_method.startswith("hybrid"):
                parts = []
                if result.dense_rank:
                    parts.append(f"Dense: #{result.dense_rank} ({result.dense_score:.3f})")
                if result.sparse_rank:
                    parts.append(f"Sparse: #{result.sparse_rank} ({result.sparse_score:.3f})")
                if parts:
                    print(f"   Sources: {' | '.join(parts)}")

            # Content preview
            preview = result.content[:200].replace('\n', ' ')
            print(f"   {preview}...")
            print()

        # Export if requested
        if args.output:
            output_path = Path(args.output) / f"{args.strategy}_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"[EXPORT] Results saved to {output_path}")


def run_batch_queries(queries_file, retrievers, config, args):
    """Run batch comparison on multiple queries."""
    # Load queries
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)

    print(f"[BATCH] Loaded {len(queries)} queries from {queries_file}")

    # Run comparison
    comparator = RetrievalComparator(retrievers)
    results = comparator.batch_compare(
        queries,
        config=config,
        output_dir=Path(args.output) if args.output else None
    )

    # Print summary statistics
    comparator.print_summary_stats(results)


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare retrieval strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Query input
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query', type=str, help='Single query to search')
    query_group.add_argument('--queries', type=str, help='File with queries (one per line)')

    # Strategy selection
    parser.add_argument(
        '--strategy',
        choices=['dense', 'sparse', 'hybrid'],
        default='hybrid',
        help='Retrieval strategy to use (default: hybrid)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all strategies side-by-side'
    )

    # Fusion configuration
    parser.add_argument(
        '--fusion',
        choices=['rrf', 'weighted'],
        default='rrf',
        help='Fusion method for hybrid retrieval (default: rrf)'
    )

    # Search configuration
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return (default: 10)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum score threshold (default: 0.0)')
    parser.add_argument('--rrf-k', type=int, default=60, help='RRF constant (default: 60)')
    parser.add_argument('--dense-weight', type=float, default=0.5, help='Dense weight for weighted fusion (default: 0.5)')
    parser.add_argument('--sparse-weight', type=float, default=0.5, help='Sparse weight for weighted fusion (default: 0.5)')
    parser.add_argument('--dense-top-k', type=int, default=50, help='Top-k for dense retrieval before fusion (default: 50)')
    parser.add_argument('--sparse-top-k', type=int, default=50, help='Top-k for sparse retrieval before fusion (default: 50)')

    # Data sources
    parser.add_argument('--collection', type=str, default='sevanbot_openai-small', help='Qdrant collection name')
    parser.add_argument('--qdrant-url', type=str, default=None, help='Qdrant URL (None for local storage at .qdrant/)')
    parser.add_argument('--qdrant-path', type=str, default='.qdrant', help='Local Qdrant storage path (used if --qdrant-url not set)')
    parser.add_argument('--chunks-file', type=str, default='chunks_contextual.jsonl', help='Path to chunks file')
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--no-stemming', dest='use_stemming', action='store_false', help='Disable Turkish stemming')

    # Output
    parser.add_argument('--output', type=str, help='Output directory for results (JSON)')

    args = parser.parse_args()

    # Validate
    if args.strategy != 'hybrid' and args.fusion != 'rrf':
        print("[WARNING] --fusion is only used with --strategy hybrid")

    # Create retrievers
    retrievers = create_retrievers(args)

    # Create config
    config = create_config(args)

    print(f"\n[CONFIG] Search configuration:")
    print(f"  top_k: {config.top_k}")
    print(f"  rrf_k: {config.rrf_k}")
    print(f"  dense_weight: {config.dense_weight}")
    print(f"  sparse_weight: {config.sparse_weight}")
    print()

    # Run queries
    if args.query:
        run_single_query(args.query, retrievers, config, args)
    elif args.queries:
        run_batch_queries(args.queries, retrievers, config, args)


if __name__ == '__main__':
    main()
