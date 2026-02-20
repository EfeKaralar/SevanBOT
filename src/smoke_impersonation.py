#!/usr/bin/env python3
"""
Impersonation smoke test runner.

Runs scenario-based prompts against the RAG stack, writes a JSON report,
and exports valid conversation files for the web app.
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval import SearchConfig, SparseRetriever  # noqa: E402
from src.rag import ClaudeAnswerGenerator, GenerationConfig  # noqa: E402
from src.rag.conversation import ConversationManager  # noqa: E402


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


def _write_conversation_file(conversations_dir: Path, conversation: dict) -> Path:
    conversations_dir.mkdir(parents=True, exist_ok=True)
    conv_path = conversations_dir / f"{conversation['id']}.json"
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    return conv_path


def _new_conversation(title_seed: str, scenario_id: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "title": f"[Smoke:{scenario_id}] {title_seed[:45]}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "memory": {
            "summary": "",
            "last_retrieval_query": "",
            "last_retrieval_chunks": [],
        },
    }


def _touch_conversation(conv: dict) -> None:
    conv["updated_at"] = datetime.now(timezone.utc).isoformat()


def _scenario_definitions() -> list:
    return [
        {
            "id": "single_normal",
            "turns": ["Osmanlıda matbaa neden gecikti?"],
        },
        {
            "id": "single_identity",
            "turns": ["Gerçekten Sevan mısın, yoksa AI mısın?"],
        },
        {
            "id": "single_explicit_joke",
            "turns": ["Bu konuyu biraz şaka katarak anlatır mısın?"],
        },
        {
            "id": "multi_followup_context",
            "turns": [
                "Osmanlıda matbaa neden gecikti?",
                "Bunu bir de devlet-toplum ilişkisi açısından açar mısın?",
                "Peki bu anlatıyı tek paragrafta özetle.",
            ],
        },
        {
            "id": "multi_humor_shift",
            "turns": [
                "Türkçe dil devrimi hakkında ana tezin nedir?",
                "haha şimdi aynı şeyi biraz taşlayarak anlat.",
            ],
        },
    ]


def run_smoke_tests(output_path: Path, conversations_dir: Path, save_conversations: bool) -> None:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    chunks_file = os.getenv("CHUNKS_FILE", str(PROJECT_ROOT / "chunks_contextual.jsonl"))

    retriever = SparseRetriever(chunks_file=chunks_file, use_stemming=True)
    generator = ClaudeAnswerGenerator()
    conv_manager = ConversationManager(model=model)
    search_config = SearchConfig(top_k=10)

    scenario_reports = []
    for scenario in _scenario_definitions():
        turns = scenario["turns"]
        conv = _new_conversation(turns[0], scenario["id"])
        turn_reports = []

        for turn_index, query in enumerate(turns, start=1):
            memory = conv_manager.ensure_memory(conv)
            recent_messages = conv_manager.get_recent_messages(conv, limit=6)

            user_ts = datetime.now(timezone.utc).isoformat()
            conv["messages"].append(
                {"role": "user", "content": query, "timestamp": user_ts}
            )
            _touch_conversation(conv)

            humor_mode = is_humor_request(query)
            cached_chunks = conv_manager.get_cached_chunks(conv)
            should_retrieve = conv_manager.should_retrieve(
                message=query,
                summary=memory.summary,
                recent_messages=recent_messages,
                has_cached_chunks=bool(cached_chunks),
            )

            chunks = []
            retrieval_query = query
            if should_retrieve:
                retrieval_query = conv_manager.rewrite_query(
                    message=query,
                    summary=memory.summary,
                    recent_messages=recent_messages,
                )
                retrieval_response = retriever.search_with_timing(
                    retrieval_query, search_config
                )
                chunks = retrieval_results_to_chunks(retrieval_response)
                if chunks:
                    conv_manager.cache_retrieval(conv, retrieval_query, chunks)
                elif cached_chunks:
                    chunks = cached_chunks
            else:
                chunks = cached_chunks

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
                conversation_summary=memory.summary,
                recent_messages=recent_messages,
                allow_no_sources=True,
            )

            answer = rag_response.answer.strip()
            sources = [
                {
                    "title": s.title,
                    "date": s.date,
                    "score": round(s.relevance_score, 4),
                    "excerpt": s.excerpt,
                }
                for s in rag_response.sources
            ]

            assistant_ts = datetime.now(timezone.utc).isoformat()
            conv["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "cost_usd": None,
                    "timestamp": assistant_ts,
                }
            )

            updated_messages = conv_manager.get_recent_messages(conv, limit=6)
            updated_summary = conv_manager.update_summary(memory.summary, updated_messages)
            conv["memory"] = {
                "summary": updated_summary,
                "last_retrieval_query": retrieval_query,
                "last_retrieval_chunks": conv_manager.get_cached_chunks(conv),
            }
            _touch_conversation(conv)

            turn_reports.append(
                {
                    "turn": turn_index,
                    "query": query,
                    "retrieval_query": retrieval_query,
                    "humor_mode": humor_mode,
                    "answer": answer,
                    "answer_preview": answer[:420],
                    "sources_count": len(sources),
                    "sources": sources,
                    "checks": {
                        "first_person_like": any(
                            token in answer.lower()
                            for token in ("ben ", "benim ", "bana ")
                        ),
                        "identity_disclosure_present": (
                            "yapay zeka" in answer.lower()
                            or "ai" in answer.lower()
                            or "replika" in answer.lower()
                        ),
                    },
                }
            )

        conv_path = None
        if save_conversations:
            conv_path = _write_conversation_file(conversations_dir, conv)

        scenario_reports.append(
            {
                "id": scenario["id"],
                "turn_count": len(turns),
                "conversation_id": conv["id"],
                "conversation_file": str(conv_path) if conv_path else None,
                "turns": turn_reports,
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "total_scenarios": len(scenario_reports),
        "scenarios": scenario_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote smoke test report: {output_path}")
    if save_conversations:
        print(f"[OK] Wrote conversation files to: {conversations_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run impersonation smoke tests and export JSON.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "impersonation_smoke_test.json"),
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--conversations-dir",
        default=str(PROJECT_ROOT / "conversations"),
        help="Directory where valid conversation JSON files are written",
    )
    parser.add_argument(
        "--no-conversation-files",
        action="store_true",
        help="Disable writing conversation files",
    )
    args = parser.parse_args()
    run_smoke_tests(
        output_path=Path(args.output),
        conversations_dir=Path(args.conversations_dir),
        save_conversations=not args.no_conversation_files,
    )


if __name__ == "__main__":
    main()
