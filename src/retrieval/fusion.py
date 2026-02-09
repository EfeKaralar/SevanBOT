"""
Fusion strategies for combining multiple ranked lists.

This module implements different methods for merging results from
dense and sparse retrievers.
"""

from typing import List, Dict
from collections import defaultdict

from .base import BaseFusion, RetrievalResult, SearchConfig


class RRFFusion(BaseFusion):
    """
    Reciprocal Rank Fusion (RRF).

    Research-backed method that combines rankings by summing reciprocal ranks.
    Robust and parameter-free (except k constant).

    Formula: score(d) = sum over rankers of 1/(k + rank(d))

    Reference: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet
    and individual systems in a larger scale experiment"
    """

    def fuse(
        self,
        results_lists: List[List[RetrievalResult]],
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """
        Combine ranked lists using RRF.

        Args:
            results_lists: List of ranked result lists
            config: Configuration (uses config.rrf_k, default 60)

        Returns:
            Fused ranking
        """
        k = config.rrf_k

        # Track fused scores by chunk_id
        fused_scores: Dict[str, Dict] = defaultdict(lambda: {
            'rrf_score': 0.0,
            'result': None,
            'sources': []
        })

        # Process each ranked list
        for list_idx, results in enumerate(results_lists):
            source_name = results[0].retrieval_method if results else f"source_{list_idx}"

            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # Add RRF score contribution
                rrf_contribution = 1.0 / (k + rank)
                fused_scores[chunk_id]['rrf_score'] += rrf_contribution

                # Track which sources contributed
                fused_scores[chunk_id]['sources'].append({
                    'method': source_name,
                    'rank': rank,
                    'score': result.score
                })

                # Keep the result object (from first occurrence)
                if fused_scores[chunk_id]['result'] is None:
                    fused_scores[chunk_id]['result'] = result

        # Convert to list and create final results
        fused_results = []
        for chunk_id, data in fused_scores.items():
            result = data['result']

            # Update with fused score
            result.score = data['rrf_score']
            result.retrieval_method = "hybrid_rrf"

            # Preserve original scores for analysis
            for source in data['sources']:
                if source['method'] == 'dense':
                    result.dense_score = source['score']
                    result.dense_rank = source['rank']
                elif source['method'] == 'sparse':
                    result.sparse_score = source['score']
                    result.sparse_rank = source['rank']

            fused_results.append(result)

        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    def get_fusion_name(self) -> str:
        """Return the name of this fusion strategy."""
        return "rrf"


class WeightedFusion(BaseFusion):
    """
    Weighted score fusion.

    Combines normalized scores using configurable weights.
    Simpler but requires score normalization and weight tuning.

    Formula: score(d) = w_dense * norm_dense(d) + w_sparse * norm_sparse(d)
    """

    def fuse(
        self,
        results_lists: List[List[RetrievalResult]],
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """
        Combine results using weighted score averaging.

        Args:
            results_lists: List of ranked result lists
            config: Configuration (uses dense_weight and sparse_weight)

        Returns:
            Fused ranking
        """
        # Normalize scores for each list (min-max normalization)
        normalized_lists = []
        for results in results_lists:
            if not results:
                normalized_lists.append([])
                continue

            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                # All scores identical, use 1.0
                for r in results:
                    r.score = 1.0
            else:
                # Min-max normalize to [0, 1]
                for r in results:
                    r.score = (r.score - min_score) / score_range

            normalized_lists.append(results)

        # Combine using weights
        weights = [config.dense_weight, config.sparse_weight]

        fused_scores: Dict[str, Dict] = defaultdict(lambda: {
            'weighted_score': 0.0,
            'result': None,
            'sources': []
        })

        for results, weight in zip(normalized_lists, weights):
            source_name = results[0].retrieval_method if results else "unknown"

            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # Add weighted score contribution
                weighted_contribution = weight * result.score
                fused_scores[chunk_id]['weighted_score'] += weighted_contribution

                # Track sources
                fused_scores[chunk_id]['sources'].append({
                    'method': source_name,
                    'rank': rank,
                    'normalized_score': result.score
                })

                if fused_scores[chunk_id]['result'] is None:
                    fused_scores[chunk_id]['result'] = result

        # Convert to list
        fused_results = []
        for chunk_id, data in fused_scores.items():
            result = data['result']
            result.score = data['weighted_score']
            result.retrieval_method = "hybrid_weighted"

            # Preserve original rankings
            for source in data['sources']:
                if source['method'] == 'dense':
                    result.dense_rank = source['rank']
                elif source['method'] == 'sparse':
                    result.sparse_rank = source['rank']

            fused_results.append(result)

        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    def get_fusion_name(self) -> str:
        """Return the name of this fusion strategy."""
        return "weighted"
