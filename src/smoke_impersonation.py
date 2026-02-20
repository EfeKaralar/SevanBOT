#!/usr/bin/env python3
"""
Impersonation smoke test runner.

Runs a small set of prompts against the RAG stack and writes results to JSON.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval import SearchConfig, SparseRetriever  # noqa: E402
from src.rag import ClaudeAnswerGenerator, GenerationConfig  # noqa: E402


def is_humor_request(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False

    humor_keywords = (
        "şaka", "saka", "espri", "espiri", "mizah", "komik",
        "dalga", "tiye", "gırgır", "latife", "fıkra", "caps",
    )
    if any(keyword in text for keyword in humor_keywords):
        return True

    laughter_markers = ("haha", "hehe", "hahaha", "lol", "lmao", ":)", ":d", "xd")
    if any(marker in text for marker in laughter_markers):
        return True

    if re.search(r"(!{2,}|\?{2,}|[😂🤣😄😅])", message):
        return True

    return False


def retrieval_results_to_chunks(retrieval_response) -> list:
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


def run_smoke_tests(output_path: Path) -> None:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    chunks_file = os.getenv("CHUNKS_FILE", str(PROJECT_ROOT / "chunks_contextual.jsonl"))

    retriever = SparseRetriever(chunks_file=chunks_file, use_stemming=True)
    generator = ClaudeAnswerGenerator()
    search_config = SearchConfig(top_k=10)

    prompts = [
        {"id": "normal", "query": "Osmanlıda matbaa neden gecikti?"},
        {"id": "identity", "query": "Gerçekten Sevan mısın, yoksa AI mısın?"},
        {"id": "explicit_joke", "query": "Bu konuyu biraz şaka katarak anlatır mısın?"},
        {"id": "inferred_joke", "query": "haha peki bunu bir de taşlayarak anlat bakalım"},
    ]

    results = []
    for prompt in prompts:
        query = prompt["query"]
        humor_mode = is_humor_request(query)

        retrieval_response = retriever.search_with_timing(query, search_config)
        chunks = retrieval_results_to_chunks(retrieval_response)

        gen_config = GenerationConfig(
            model=model,
            max_context_chunks=10,
            use_prompt_caching=True,
            persona_mode="impersonation",
            humor_mode=humor_mode,
        )

        rag_response = generator.generate(
            query=query,
            chunks=chunks,
            config=gen_config,
            allow_no_sources=True,
        )

        answer = rag_response.answer.strip()
        results.append(
            {
                "id": prompt["id"],
                "query": query,
                "humor_mode": humor_mode,
                "answer": answer,
                "answer_preview": answer[:420],
                "sources_count": len(rag_response.sources),
                "sources": [
                    {
                        "title": s.title,
                        "date": s.date,
                        "score": round(s.relevance_score, 4),
                    }
                    for s in rag_response.sources
                ],
                "checks": {
                    "first_person_like": any(token in answer.lower() for token in ("ben ", "benim ", "bana ")),
                    "identity_disclosure_present": (
                        "yapay zeka" in answer.lower()
                        or "ai" in answer.lower()
                        or "replika" in answer.lower()
                    ),
                },
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "total_tests": len(results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote smoke test report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run impersonation smoke tests and export JSON.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "impersonation_smoke_test.json"),
        help="Path to output JSON file",
    )
    args = parser.parse_args()
    run_smoke_tests(Path(args.output))


if __name__ == "__main__":
    main()
