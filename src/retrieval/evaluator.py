"""
Evaluation and comparison tools for retrieval strategies.

This module provides utilities to compare different retrieval methods,
analyze rankings, and compute metrics.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .base import BaseRetriever, SearchConfig, RetrievalResponse


@dataclass
class ComparisonResult:
    """Results from comparing multiple retrieval strategies."""
    query: str
    responses: Dict[str, RetrievalResponse]  # strategy_name -> response

    def print_summary(self):
        """Print human-readable comparison summary."""
        print(f"\n{'='*80}")
        print(f"QUERY: {self.query}")
        print(f"{'='*80}\n")

        for strategy_name, response in self.responses.items():
            print(f"--- {strategy_name.upper()} ({response.strategy.value}) ---")
            print(f"Time: {response.total_time_ms:.1f}ms", end="")
            if response.dense_time_ms > 0:
                print(f" (dense: {response.dense_time_ms:.1f}ms, "
                      f"sparse: {response.sparse_time_ms:.1f}ms, "
                      f"fusion: {response.fusion_time_ms:.1f}ms)", end="")
            print(f"\nCandidates: {response.total_candidates}")
            print(f"Results: {len(response.results)}\n")

            # Show top 5 results
            for i, result in enumerate(response.results[:5], 1):
                print(f"{i}. [Score: {result.score:.4f}]")
                if result.metadata:
                    title = result.metadata.get('title', 'No title')
                    print(f"   Title: {title}")

                # Show provenance for hybrid results
                if result.retrieval_method.startswith("hybrid"):
                    parts = []
                    if result.dense_rank:
                        parts.append(f"Dense: #{result.dense_rank} ({result.dense_score:.3f})")
                    if result.sparse_rank:
                        parts.append(f"Sparse: #{result.sparse_rank} ({result.sparse_score:.3f})")
                    if parts:
                        print(f"   Sources: {' | '.join(parts)}")

                # Show content preview
                preview = result.content[:150].replace('\n', ' ')
                print(f"   {preview}...")
                print()

            print()

    def analyze_overlap(self) -> Dict[str, Any]:
        """
        Analyze overlap between different retrieval strategies.

        Returns:
            Dictionary with overlap statistics
        """
        if len(self.responses) < 2:
            return {}

        strategies = list(self.responses.keys())
        results_by_strategy = {
            name: set(r.chunk_id for r in resp.results)
            for name, resp in self.responses.items()
        }

        analysis = {
            "total_unique_chunks": len(set.union(*results_by_strategy.values())),
            "pairwise_overlap": {}
        }

        # Compute pairwise overlaps
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                chunks1 = results_by_strategy[strat1]
                chunks2 = results_by_strategy[strat2]

                intersection = len(chunks1 & chunks2)
                union = len(chunks1 | chunks2)
                jaccard = intersection / union if union > 0 else 0

                analysis["pairwise_overlap"][f"{strat1}_vs_{strat2}"] = {
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "only_in_first": len(chunks1 - chunks2),
                    "only_in_second": len(chunks2 - chunks1),
                }

        return analysis

    def export_to_json(self, output_path: Path):
        """Export comparison to JSON for detailed analysis."""
        data = {
            "query": self.query,
            "strategies": {
                name: resp.to_dict()
                for name, resp in self.responses.items()
            },
            "overlap_analysis": self.analyze_overlap()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[EXPORT] Comparison saved to {output_path}")


class RetrievalComparator:
    """
    Tool for comparing different retrieval strategies.

    Enables side-by-side comparison of dense, sparse, and hybrid search.
    """

    def __init__(self, retrievers: Dict[str, BaseRetriever]):
        """
        Initialize comparator.

        Args:
            retrievers: Dictionary mapping strategy names to retriever instances
                       e.g., {"dense": DenseRetriever(...), "sparse": SparseRetriever(...)}
        """
        self.retrievers = retrievers

    def compare(
        self,
        query: str,
        config: SearchConfig = None
    ) -> ComparisonResult:
        """
        Execute query across all retrievers and compare results.

        Args:
            query: User query string
            config: Search configuration (uses defaults if None)

        Returns:
            ComparisonResult with responses from all retrievers
        """
        if config is None:
            config = SearchConfig()

        print(f"\n[COMPARE] Running query across {len(self.retrievers)} retrievers...")
        print(f"[COMPARE] Query: '{query}'")
        print(f"[COMPARE] Config: top_k={config.top_k}, rrf_k={config.rrf_k}")

        responses = {}
        for name, retriever in self.retrievers.items():
            print(f"[COMPARE] Executing {name}...", end=" ")
            response = retriever.search_with_timing(query, config)
            responses[name] = response
            print(f"✓ ({response.total_time_ms:.1f}ms, {len(response.results)} results)")

        return ComparisonResult(query=query, responses=responses)

    def batch_compare(
        self,
        queries: List[str],
        config: SearchConfig = None,
        output_dir: Path = None
    ) -> List[ComparisonResult]:
        """
        Run comparison on multiple queries.

        Args:
            queries: List of query strings
            config: Search configuration
            output_dir: Optional directory to save results

        Returns:
            List of ComparisonResult objects
        """
        results = []

        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}/{len(queries)}")
            print(f"{'='*80}")

            result = self.compare(query, config)
            results.append(result)

            # Export if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"query_{i:03d}.json"
                result.export_to_json(output_file)

        return results

    def print_summary_stats(self, results: List[ComparisonResult]):
        """
        Print aggregate statistics across multiple queries.

        Args:
            results: List of comparison results
        """
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS ({len(results)} queries)")
        print(f"{'='*80}\n")

        # Aggregate timing stats
        timing_stats = {name: [] for name in self.retrievers.keys()}
        for result in results:
            for name, response in result.responses.items():
                timing_stats[name].append(response.total_time_ms)

        print("Average Response Times:")
        for name, times in timing_stats.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"  {name:15s}: {avg_time:6.1f}ms (min: {min_time:.1f}ms, max: {max_time:.1f}ms)")

        # Aggregate overlap stats (if multiple strategies)
        if len(self.retrievers) >= 2:
            print("\nAverage Overlap (Jaccard Similarity):")
            all_overlaps = {}
            for result in results:
                overlap_analysis = result.analyze_overlap()
                for pair, stats in overlap_analysis.get("pairwise_overlap", {}).items():
                    if pair not in all_overlaps:
                        all_overlaps[pair] = []
                    all_overlaps[pair].append(stats["jaccard"])

            for pair, jaccards in all_overlaps.items():
                avg_jaccard = sum(jaccards) / len(jaccards)
                print(f"  {pair:30s}: {avg_jaccard:.3f}")
