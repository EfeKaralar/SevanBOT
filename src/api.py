#!/usr/bin/env python3
"""
SevanBot Chat Web App — FastAPI backend.

Serves the single-page chat interface and provides REST + SSE endpoints
for conversation management and RAG-powered answer generation.

Usage (from project root):
    python3 src/api.py
    # → http://localhost:8000
"""

import asyncio
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import dotenv
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — mirrors answer_rag.py pattern
# ---------------------------------------------------------------------------
# Add project root to sys.path so "from src.retrieval import ..." works
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval import (  # noqa: E402
    DenseRetriever,
    HybridRetriever,
    RRFFusion,
    SearchConfig,
    SparseRetriever,
)
from src.rag import ClaudeAnswerGenerator, GenerationConfig  # noqa: E402
from src.rag.conversation import ConversationManager  # noqa: E402
from src.rag.retrieval_planner import RetrievalPlanner, TIER_CONFIG  # noqa: E402
from src.rag.verbosity import derive_response_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", str(PROJECT_ROOT / "conversations")))
STATIC_DIR = PROJECT_ROOT / "static"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sevanbot_openai-small")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", str(PROJECT_ROOT / "chunks_contextual.jsonl"))
QDRANT_PATH = os.getenv("QDRANT_PATH", str(PROJECT_ROOT / ".qdrant"))
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


GENERATION_MODEL = os.getenv("GENERATION_MODEL", ANTHROPIC_MODEL)
GENERATION_TEMPERATURE = _env_float("GENERATION_TEMPERATURE", 0.1)
GENERATION_MAX_CONTEXT_CHUNKS = _env_int("GENERATION_MAX_CONTEXT_CHUNKS", 10)
CONVERSATION_HELPER_MODEL = os.getenv(
    "CONVERSATION_HELPER_MODEL",
    "claude-3-5-haiku-20241022",
)
CONVERSATION_HELPER_MAX_TOKENS = _env_int("CONVERSATION_HELPER_MAX_TOKENS", 384)

# ---------------------------------------------------------------------------
# Global retriever / generator (initialized at startup)
# ---------------------------------------------------------------------------
_dense_retriever: Optional[DenseRetriever] = None
_sparse_retriever: Optional[SparseRetriever] = None
_retriever: Optional[HybridRetriever] = None
_generator: Optional[ClaudeAnswerGenerator] = None
_conversation_manager: Optional[ConversationManager] = None
_retrieval_planner: Optional[RetrievalPlanner] = None


def get_retriever(strategy: str = "hybrid"):
    if strategy == "dense":
        if _dense_retriever is None:
            raise RuntimeError("Dense retriever not initialized")
        return _dense_retriever
    if strategy == "sparse":
        if _sparse_retriever is None:
            raise RuntimeError("Sparse retriever not initialized")
        return _sparse_retriever
    if _retriever is None:
        raise RuntimeError("Hybrid retriever not initialized")
    return _retriever


def get_generator() -> ClaudeAnswerGenerator:
    if _generator is None:
        raise RuntimeError("Generator not initialized")
    return _generator


def get_conversation_manager() -> ConversationManager:
    if _conversation_manager is None:
        raise RuntimeError("Conversation manager not initialized")
    return _conversation_manager


def get_retrieval_planner() -> RetrievalPlanner:
    if _retrieval_planner is None:
        raise RuntimeError("Retrieval planner not initialized")
    return _retrieval_planner


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="SevanBot", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def startup_event():
    global _dense_retriever, _sparse_retriever, _retriever, _generator, _conversation_manager, _retrieval_planner

    print("[STARTUP] Initializing SevanBot...")

    if not APP_PASSWORD:
        print("[WARNING] APP_PASSWORD is not set in .env — the app will be accessible without a password.")

    CONVERSATIONS_DIR.mkdir(exist_ok=True)

    print("[STARTUP] Loading dense retriever...")
    _dense_retriever = DenseRetriever(
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        qdrant_path=QDRANT_PATH,
        qdrant_api_key=QDRANT_API_KEY,
        embedding_model="text-embedding-3-small",
    )

    print("[STARTUP] Loading sparse retriever...")
    _sparse_retriever = SparseRetriever(
        chunks_file=CHUNKS_FILE,
        use_stemming=True,
    )

    print("[STARTUP] Creating hybrid retriever (RRF fusion)...")
    _retriever = HybridRetriever(
        dense_retriever=_dense_retriever,
        sparse_retriever=_sparse_retriever,
        fusion_strategy=RRFFusion(),
    )

    print("[STARTUP] Initializing Claude generator...")
    _generator = ClaudeAnswerGenerator()
    _conversation_manager = ConversationManager(
        model=CONVERSATION_HELPER_MODEL,
        max_tokens=CONVERSATION_HELPER_MAX_TOKENS,
    )

    print("[STARTUP] Initializing retrieval planner...")
    # Enable adaptive retrieval (set ADAPTIVE_RETRIEVAL_ENABLED=false to disable)
    planner_enabled = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").lower() != "false"
    _retrieval_planner = RetrievalPlanner(enabled=planner_enabled)
    if not planner_enabled:
        print("[STARTUP] Adaptive retrieval disabled, using default tier")

    print("[STARTUP] SevanBot ready at http://localhost:8000")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
security = HTTPBearer(auto_error=False)


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Validate bearer token against APP_PASSWORD."""
    if not APP_PASSWORD:
        # No password configured — allow all requests
        return
    if credentials is None or credentials.credentials != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Conversation storage helpers
# ---------------------------------------------------------------------------

def _conv_path(conv_id: str) -> Path:
    return CONVERSATIONS_DIR / f"{conv_id}.json"


def list_conversations() -> list:
    """Return all conversations sorted newest first (summary only)."""
    convs = []
    for path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            convs.append({
                "id": data["id"],
                "title": data.get("title", "Yeni Sohbet"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    convs.sort(key=lambda c: c["updated_at"], reverse=True)
    return convs


def load_conversation(conv_id: str) -> dict:
    """Load full conversation by id. Raises 404 if not found."""
    path = _conv_path(conv_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_conversation(conv: dict) -> None:
    """Persist conversation to disk."""
    path = _conv_path(conv["id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2, ensure_ascii=False)


def delete_conversation(conv_id: str) -> None:
    """Delete conversation file. Raises 404 if not found."""
    path = _conv_path(conv_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    path.unlink()


def _new_conversation(first_message: str) -> dict:
    """Create a fresh conversation dict."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "title": first_message[:60],
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def _touch_conversation(conv: dict) -> None:
    """Update the updated_at timestamp."""
    conv["updated_at"] = datetime.now(timezone.utc).isoformat()


def _is_humor_request(message: str) -> bool:
    """Heuristic detector for joke-friendly user intent."""
    text = (message or "").strip().lower()
    if not text:
        return False

    humor_keywords = (
        "şaka", "saka", "espri", "espiri", "mizah", "komik",
        "dalga", "tiye", "gırgır", "latife", "fıkra", "caps",
    )
    if any(keyword in text for keyword in humor_keywords):
        return True

    laughter_markers = (
        "haha", "hehe", "hahaha", "lol", "lmao", ":)", ":d", "xd",
    )
    if any(marker in text for marker in laughter_markers):
        return True

    # Lightweight punctuation cue for playful prompts.
    if re.search(r"(!{2,}|\?{2,}|[😂🤣😄😅])", message):
        return True

    return False


# ---------------------------------------------------------------------------
# Async streaming helper
# ---------------------------------------------------------------------------

async def _stream_tokens(
    gen,
    query: str,
    chunks: list,
    config,
    conversation_summary: Optional[str] = None,
    recent_messages: Optional[list] = None,
    allow_no_sources: bool = False,
) -> AsyncIterator[str]:
    """
    Bridge between the synchronous Anthropic SDK streaming iterator and the
    asyncio event loop.

    The sync iterator blocks the OS thread between tokens (httpx I/O). Running
    it directly inside an async generator would block the event loop, preventing
    uvicorn from flushing SSE chunks to the client until generation completes.

    Solution: run the sync iterator in a background thread and ferry each token
    back to the event loop through an asyncio.Queue.  Each `await queue.get()`
    yields control to the event loop so uvicorn can flush the preceding chunk
    before waiting for the next token.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
                for token in gen.generate_streaming(
                    query,
                    chunks,
                    config,
                    conversation_summary=conversation_summary,
                    recent_messages=recent_messages,
                    allow_no_sources=allow_no_sources,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    strategy: str = "hybrid"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/health")
async def healthcheck():
    """Railway healthcheck endpoint."""
    return {"ok": True}


@app.get("/api/conversations", dependencies=[Depends(require_auth)])
async def get_conversations():
    """List all conversations (summary only)."""
    return list_conversations()


@app.get("/api/conversations/{conv_id}", dependencies=[Depends(require_auth)])
async def get_conversation(conv_id: str):
    """Get full conversation with all messages."""
    return load_conversation(conv_id)


@app.delete("/api/conversations/{conv_id}", dependencies=[Depends(require_auth)])
async def remove_conversation(conv_id: str):
    """Delete a conversation."""
    delete_conversation(conv_id)
    return {"ok": True}


@app.post("/api/chat", dependencies=[Depends(require_auth)])
async def chat(req: ChatRequest):
    """
    Handle a chat message and stream the answer as SSE.

    SSE event types:
      token  — {"text": "..."}   (one per token during generation)
      done   — {"conversation_id": "...", "sources": [...], "cost_usd": 0.002}
      error  — {"message": "..."}
    """
    strategy = req.strategy if req.strategy in ("dense", "sparse", "hybrid") else "hybrid"
    retriever = get_retriever(strategy)
    generator = get_generator()
    conv_manager = get_conversation_manager()
    planner = get_retrieval_planner()

    # Load or create conversation
    if req.conversation_id:
        conv = load_conversation(req.conversation_id)
    else:
        conv = _new_conversation(req.message)

    # Ensure memory exists and collect recent context before adding new turn
    memory = conv_manager.ensure_memory(conv)
    recent_messages = conv_manager.get_recent_messages(conv, limit=6)

    # Append user message
    now = datetime.now(timezone.utc).isoformat()
    conv["messages"].append({"role": "user", "content": req.message, "timestamp": now})
    _touch_conversation(conv)
    save_conversation(conv)

    async def event_stream():
        full_answer = []
        humor_mode = _is_humor_request(req.message)

        try:
            # Use LLM-based planner to decide retrieval tier
            cached_chunks = conv_manager.get_cached_chunks(conv)
            plan = planner.plan(
                query=req.message,
                conversation_summary=memory.summary,
                cached_chunks=cached_chunks,
                recent_messages=recent_messages,
            )
            print(f"[Planner] tier={plan.tier}, reasoning={plan.reasoning[:80]}...")

            chunks = []
            retrieval_query = plan.rewritten_query or req.message
            if plan.should_retrieve:
                # Get tier-specific config
                tier_config = TIER_CONFIG[plan.tier]
                search_config = SearchConfig(
                    top_k=tier_config["top_k"],
                    rrf_k=60,
                    dense_top_k=50,
                    sparse_top_k=50,
                )
                retrieval_response = retriever.search_with_timing(retrieval_query, search_config)
                chunks = [
                    {
                        "chunk_id": r.chunk_id,
                        "text": r.content,
                        "title": r.metadata.get("title", ""),
                        "date": r.metadata.get("date", ""),
                        "score": r.score,
                    }
                    for r in retrieval_response.results
                ]
                if chunks:
                    conv_manager.cache_retrieval(conv, retrieval_query, chunks)
                    save_conversation(conv)
                elif cached_chunks:
                    # Fallback to cached if retrieval returned nothing
                    chunks = cached_chunks
            else:
                # No retrieval needed, use cached chunks
                chunks = cached_chunks

            response_settings = derive_response_settings(
                query=req.message,
                chunks=chunks,
                recent_messages=recent_messages,
            )

            # Generate answer with streaming using tier + adaptive verbosity.
            tier_config = TIER_CONFIG[plan.tier]
            tier_max_chunks = tier_config["max_chunks"] if plan.should_retrieve else len(chunks)
            max_context_chunks = max(1, tier_max_chunks)
            max_context_chunks = min(max_context_chunks, GENERATION_MAX_CONTEXT_CHUNKS)
            gen_config = GenerationConfig(
                model=GENERATION_MODEL,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=response_settings.max_tokens,
                max_context_chunks=max_context_chunks,
                response_instruction=response_settings.response_instruction,
                use_prompt_caching=True,
                persona_mode="impersonation",
                humor_mode=humor_mode,
            )

            async for token in _stream_tokens(
                generator,
                req.message,
                chunks,
                gen_config,
                conversation_summary=memory.summary,
                recent_messages=recent_messages,
                allow_no_sources=True,
            ):
                full_answer.append(token)
                yield "event: token\ndata: " + json.dumps({"text": token}) + "\n\n"

            # Build final response data
            answer_text = "".join(full_answer)

            # Build sources from chunks used in generation
            sources = [
                {
                    "title": c["title"],
                    "date": c["date"],
                    "score": round(c["score"], 4),
                    "excerpt": c["text"][:200],
                }
                for c in (chunks or [])[:gen_config.max_context_chunks]
            ]

            # Save assistant message (without exact token cost since we streamed)
            msg_timestamp = datetime.now(timezone.utc).isoformat()
            conv["messages"].append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "cost_usd": None,  # Cost not available in streaming mode
                "timestamp": msg_timestamp,
            })
            # Update conversation summary after assistant response
            updated_messages = conv_manager.get_recent_messages(conv, limit=6)
            updated_summary = conv_manager.update_summary(memory.summary, updated_messages)
            conv["memory"] = {
                "summary": updated_summary,
                "last_retrieval_query": retrieval_query,
                "last_retrieval_chunks": conv_manager.get_cached_chunks(conv),
            }
            _touch_conversation(conv)
            save_conversation(conv)

            # Send done event
            yield "event: done\ndata: " + json.dumps({
                "conversation_id": conv["id"],
                "sources": sources,
                "cost_usd": None,
            }) + "\n\n"

        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        app_dir=str(Path(__file__).parent),
    )
