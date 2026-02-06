#!/usr/bin/env python3
"""
Semantic chunking for SevanBot documents (v3).

Splits markdown articles into chunks suitable for embedding:
- Primary split: paragraphs
- Merges small paragraphs until minimum size reached
- Fallback: sentences (if paragraph > max tokens)
- Percentage-based overlap between chunks (improved from v2)
- Filters noise (footnotes, images, Ottoman examples)

v3 Improvements (based on 2025 RAG research):
- Context enrichment: prepends title/date/keywords to chunk for better retrieval
- Improved metadata: prechunk_id, postchunk_id, doc_id for context expansion
- Percentage-based overlap (10-20% recommended)
- Outputs both raw text and text_for_embedding (context-enriched)
"""

import json
import re
import hashlib
import warnings
from pathlib import Path
from transformers import AutoTokenizer

# Suppress tokenizer sequence length warnings (we handle truncation ourselves)
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than")

# Directories
FORMATTED_DIR = Path(__file__).parent.parent / "formatted"
OUTPUT_FILE = Path(__file__).parent.parent / "chunks.jsonl"
OUTPUT_FILE_V2 = Path(__file__).parent.parent / "chunks_v2.jsonl"  # Old format for comparison

# Chunking parameters (adjusted based on research: 256-512 optimal)
# Note: E5 model has 512 token limit, so we leave room for context prefix + overlap
MAX_TOKENS = 400
MIN_TOKENS = 200
OVERLAP_PERCENT = 0.15  # 15% overlap (research suggests 10-20%)
EMBEDDING_MODEL_LIMIT = 512  # Hard limit for final text_for_embedding

# Ottoman text detection (special diacritics used in transliteration)
OTTOMAN_PATTERN = re.compile(r'[ˁāīūḥḍṣṭẓġʿʾ]')


def generate_doc_id(file_path: str, title: str) -> str:
    """Generate a unique document ID from file path and title."""
    content = f"{file_path}:{title or 'untitled'}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def build_context_prefix(metadata: dict) -> str:
    """
    Build a context prefix to prepend to chunk text.

    This implements the "Document Title Prepending" technique from
    contextual retrieval research - simple but effective for improving
    retrieval by providing document-level context to each chunk.
    """
    parts = []

    if metadata.get("title"):
        parts.append(f"Makale: {metadata['title']}")

    if metadata.get("date"):
        parts.append(f"Tarih: {metadata['date']}")

    if metadata.get("keywords"):
        parts.append(f"Konular: {metadata['keywords']}")

    if not parts:
        return ""

    return " | ".join(parts) + "\n\n"


def get_overlap_text(text: str, tokenizer, target_tokens: int) -> str:
    """
    Get overlap text from the end of a chunk.

    Uses percentage-based overlap instead of fixed sentence count
    for more consistent context across different chunk sizes.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    # Build overlap from end, staying under target tokens
    overlap_sentences = []
    current_tokens = 0

    for sent in reversed(sentences):
        sent_tokens = count_tokens(sent, tokenizer)
        if current_tokens + sent_tokens > target_tokens:
            break
        overlap_sentences.insert(0, sent)
        current_tokens += sent_tokens

    return " ".join(overlap_sentences)


def load_tokenizer():
    """Load the E5 model tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_markdown(file_path: Path) -> dict:
    """
    Parse markdown file and extract metadata.
    Returns dict with: title, date, source, keywords, content
    """
    text = file_path.read_text(encoding="utf-8")

    metadata = {
        "title": None,
        "date": None,
        "source": None,
        "keywords": None,
        "content": None,
        "file": str(file_path.name),
    }

    # Split on --- separator
    parts = text.split("---", 1)
    header = parts[0] if parts else text
    content = parts[1].strip() if len(parts) > 1 else ""

    # Extract title (first # heading)
    title_match = re.search(r"^#\s+(.+)$", header, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    # Extract date - various formats
    date_patterns = [
        r"^\d{1,2}\s+\w+\s+\d{4}$",  # 23 October 2022
        r"^\w{3}\s+\d{1,2},\s+\d{4}$",  # Mar 27, 2021
    ]
    for line in header.split("\n"):
        line = line.strip()
        for pattern in date_patterns:
            if re.match(pattern, line):
                metadata["date"] = line
                break

    # Extract source URL
    source_match = re.search(r"\*\*(?:Source|Kaynak):\*\*\s*(https?://\S+)", header)
    if source_match:
        metadata["source"] = source_match.group(1)

    # Extract keywords (sevan format only)
    keywords_match = re.search(r"\*\*Anahtar Kelimeler:\*\*\s*(.+)$", header, re.MULTILINE)
    if keywords_match:
        metadata["keywords"] = keywords_match.group(1).strip()

    metadata["content"] = content
    return metadata


def is_noise(text: str) -> bool:
    """Check if text is noise that should be skipped."""
    text = text.strip()

    # Empty or whitespace only
    if not text:
        return True

    # Image-only paragraphs
    if text.startswith("[![") or text.startswith("!["):
        return True

    # Footnote references like [1], [2], [3] A.g.e. sf.144
    if re.match(r'^\[\d+\]', text):
        return True

    # Very short non-content (single words, punctuation)
    if len(text) < 10:
        return True

    return False


def is_ottoman_example(text: str) -> bool:
    """Check if text is an Ottoman transliteration example to skip."""
    # Blockquotes with Ottoman diacritics
    if text.startswith(">") and OTTOMAN_PATTERN.search(text):
        return True

    # High density of Ottoman characters suggests transliteration block
    if len(text) > 100:
        ottoman_chars = len(OTTOMAN_PATTERN.findall(text))
        if ottoman_chars > 10:  # More than 10 special chars
            return True

    return False


def extract_paragraphs(content: str) -> list[str]:
    """
    Extract paragraphs from content, handling blockquotes specially.
    """
    paragraphs = []
    current_blockquote = []

    for line in content.split("\n"):
        if line.startswith(">"):
            current_blockquote.append(line)
        else:
            # Flush blockquote if we were in one
            if current_blockquote:
                blockquote_text = "\n".join(current_blockquote)
                if not is_ottoman_example(blockquote_text):
                    paragraphs.append(blockquote_text)
                current_blockquote = []

            # Regular paragraph handling
            if line.strip():
                # Check if this continues the previous paragraph
                if paragraphs and not line.startswith(("#", "*", "-", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    # Could be continuation, but for simplicity treat as new
                    pass
                paragraphs.append(line.strip())
            elif paragraphs and paragraphs[-1]:
                # Empty line - paragraph break (marker)
                paragraphs.append("")

    # Flush final blockquote
    if current_blockquote:
        blockquote_text = "\n".join(current_blockquote)
        if not is_ottoman_example(blockquote_text):
            paragraphs.append(blockquote_text)

    # Merge consecutive non-empty paragraphs that were split by single newlines
    merged = []
    current = []
    for p in paragraphs:
        if p == "":
            if current:
                merged.append("\n\n".join(current))
                current = []
        else:
            current.append(p)
    if current:
        merged.append("\n\n".join(current))

    return merged


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space and capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ"])', text)
    return [s.strip() for s in sentences if s.strip()]


def hard_split_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Force-split text into chunks under max_tokens.
    Used as fallback when sentence splitting doesn't work (e.g., long lists).
    Splits on commas, spaces, or by character count as last resort.
    """
    # Try different delimiters in order of preference
    for delimiter in [', ', ' ', '']:
        if delimiter == '':
            # Last resort: split by character count (~4 chars per token)
            char_limit = max_tokens * 3  # Conservative estimate
            chunks = []
            for i in range(0, len(text), char_limit):
                chunks.append(text[i:i + char_limit])
            return chunks

        if delimiter not in text:
            continue

        parts = text.split(delimiter)
        chunks = []
        current = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part, tokenizer)
            if current_tokens + part_tokens > max_tokens and current:
                chunks.append(delimiter.join(current))
                current = [part]
                current_tokens = part_tokens
            else:
                current.append(part)
                current_tokens += part_tokens + 1

        if current:
            chunks.append(delimiter.join(current))

        # Verify all chunks are under limit
        all_under = all(count_tokens(c, tokenizer) <= max_tokens for c in chunks)
        if all_under:
            return chunks

    return [text]  # Shouldn't reach here


def truncate_to_limit(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    # Use fewer tokens to account for encode/decode variance
    truncated = tokenizer.decode(tokens[:max_tokens - 15])
    return truncated


def get_last_sentences(text: str, n: int = 1) -> str:
    """Get last n sentences from text for overlap."""
    sentences = split_into_sentences(text)
    if len(sentences) <= n:
        return text
    return " ".join(sentences[-n:])


def chunk_document(
    metadata: dict,
    tokenizer,
    enrich_context: bool = True,
    context_mode: str = "simple",
    anthropic_client = None,
    stats_tracker = None
) -> list[dict]:
    """
    Chunk a document into pieces suitable for embedding (v3).

    Args:
        metadata: Document metadata (title, date, content, etc.)
        tokenizer: Tokenizer for counting tokens
        enrich_context: Whether to enrich chunks with context (default: True)
        context_mode: "simple" (metadata prepending) or "llm" (Claude-generated)
        anthropic_client: Anthropic client (required if context_mode="llm")
        stats_tracker: ContextGenerationStats object to track costs (optional)

    Returns list of chunk dicts with:
    - text: raw chunk text
    - text_for_embedding: context-enriched text (if enrich_context=True)
    - metadata: title, date, source, keywords, file
    - references: doc_id, chunk_id, prechunk_id, postchunk_id, total_chunks
    - context_mode: "simple" or "llm"
    - llm_context: LLM-generated context (only if context_mode="llm")
    """
    # Validate context_mode
    if context_mode not in ["simple", "llm"]:
        raise ValueError(f"context_mode must be 'simple' or 'llm', got: {context_mode}")

    if context_mode == "llm" and anthropic_client is None:
        raise ValueError("anthropic_client is required when context_mode='llm'")
    content = metadata["content"]
    if not content:
        return []

    # Extract paragraphs
    paragraphs = extract_paragraphs(content)

    # Filter noise
    paragraphs = [p for p in paragraphs if not is_noise(p) and not is_ottoman_example(p)]

    if not paragraphs:
        return []

    # Generate document ID
    doc_id = generate_doc_id(metadata["file"], metadata["title"])

    # Build chunks by merging small paragraphs
    raw_chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)

        # If single paragraph exceeds max, split by sentences
        if para_tokens > MAX_TOKENS:
            # Flush current chunk first
            if current_chunk:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph
            sentences = split_into_sentences(para)

            # If sentence split didn't help (e.g., one giant list), use hard split
            if len(sentences) == 1 and para_tokens > MAX_TOKENS:
                hard_chunks = hard_split_text(para, tokenizer, MAX_TOKENS)
                raw_chunks.extend(hard_chunks)
                continue

            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_tok = count_tokens(sent, tokenizer)

                # If single sentence is too long, hard split it
                if sent_tok > MAX_TOKENS:
                    if sent_chunk:
                        raw_chunks.append(" ".join(sent_chunk))
                        sent_chunk = []
                        sent_tokens = 0
                    hard_chunks = hard_split_text(sent, tokenizer, MAX_TOKENS)
                    raw_chunks.extend(hard_chunks)
                    continue

                if sent_tokens + sent_tok > MAX_TOKENS and sent_chunk:
                    raw_chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_tokens = 0
                sent_chunk.append(sent)
                sent_tokens += sent_tok

            if sent_chunk:
                # Don't add tiny leftover as its own chunk, carry forward
                leftover = " ".join(sent_chunk)
                if sent_tokens >= MIN_TOKENS:
                    raw_chunks.append(leftover)
                else:
                    current_chunk = [leftover]
                    current_tokens = sent_tokens
            continue

        # Would adding this exceed max?
        if current_tokens + para_tokens > MAX_TOKENS and current_chunk:
            # Only flush if we've reached minimum
            if current_tokens >= MIN_TOKENS:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                # Keep accumulating even if slightly over max
                current_chunk.append(para)
                current_tokens += para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Flush remaining
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        # If too small and we have previous chunks, merge with last
        if current_tokens < MIN_TOKENS and raw_chunks:
            raw_chunks[-1] = raw_chunks[-1] + "\n\n" + chunk_text
        else:
            raw_chunks.append(chunk_text)

    total_chunks = len(raw_chunks)

    # Build simple context prefix once (same for all chunks in this document)
    simple_context_prefix = build_context_prefix(metadata) if enrich_context else ""

    # For LLM mode, we need the full document content
    full_doc_content = metadata["content"] if context_mode == "llm" else None

    # Add percentage-based overlap and build final chunks
    final_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Calculate overlap tokens based on previous chunk size
        if i > 0:
            prev_chunk_tokens = count_tokens(raw_chunks[i-1], tokenizer)
            target_overlap_tokens = int(prev_chunk_tokens * OVERLAP_PERCENT)

            # Add overlap from previous chunk
            if target_overlap_tokens > 10:
                overlap = get_overlap_text(raw_chunks[i-1], tokenizer, target_overlap_tokens)
                if overlap:
                    chunk_text = overlap + " " + chunk_text

        # Generate chunk ID
        chunk_id = f"{doc_id}#{i}"

        # Build text_for_embedding based on context mode
        llm_context = None
        if not enrich_context:
            text_for_embed = chunk_text
        elif context_mode == "simple":
            text_for_embed = simple_context_prefix + chunk_text
        else:  # context_mode == "llm"
            # Generate LLM-based context for this chunk
            from contextual_utils import situate_context, validate_context
            import time

            try:
                llm_context, usage = situate_context(
                    full_doc_content,
                    chunk_text,
                    metadata["title"] or metadata["file"],
                    anthropic_client
                )

                # Track usage stats if provided
                if stats_tracker:
                    stats_tracker.add_usage(usage)

                # Rate limiting: small delay between chunks to avoid overwhelming API
                # Only delay if we successfully got a response
                time.sleep(0.2)  # 200ms delay = max 5 requests/sec

                # Validate the generated context
                if validate_context(llm_context, chunk_text):
                    # Prepend LLM context to chunk
                    text_for_embed = llm_context + "\n\n" + chunk_text
                else:
                    # Fallback to simple mode if validation fails
                    print(f"Warning: Invalid LLM context for chunk {i}, falling back to simple mode")
                    text_for_embed = simple_context_prefix + chunk_text
                    llm_context = None

            except Exception as e:
                # Fallback to simple mode on error
                print(f"Error generating LLM context for chunk {i}: {e}")
                print("Falling back to simple mode")
                text_for_embed = simple_context_prefix + chunk_text
                llm_context = None

        # Safety truncation: ensure we don't exceed embedding model limit
        if count_tokens(text_for_embed, tokenizer) > EMBEDDING_MODEL_LIMIT:
            text_for_embed = truncate_to_limit(text_for_embed, tokenizer, EMBEDDING_MODEL_LIMIT)

        # Build the chunk with improved metadata
        chunk = {
            # Content
            "text": chunk_text,
            "text_for_embedding": text_for_embed,

            # Document metadata
            "title": metadata["title"],
            "date": metadata["date"],
            "source": metadata["source"],
            "keywords": metadata["keywords"],
            "file": metadata["file"],

            # Chunk references (for retrieval-time context expansion)
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": i,
            "total_chunks": total_chunks,
            "prechunk_id": f"{doc_id}#{i-1}" if i > 0 else None,
            "postchunk_id": f"{doc_id}#{i+1}" if i < total_chunks - 1 else None,

            # Context mode tracking
            "context_mode": context_mode,
        }

        # Add LLM context separately if available
        if llm_context:
            chunk["llm_context"] = llm_context

        final_chunks.append(chunk)

    return final_chunks


def chunk_document_v2(metadata: dict, tokenizer) -> list[dict]:
    """
    Legacy v2 chunking (for comparison).
    Uses fixed 1-sentence overlap and no context enrichment.
    """
    content = metadata["content"]
    if not content:
        return []

    paragraphs = extract_paragraphs(content)
    paragraphs = [p for p in paragraphs if not is_noise(p) and not is_ottoman_example(p)]

    if not paragraphs:
        return []

    # Use old parameters
    old_max = 400
    old_min = 150

    raw_chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)

        if para_tokens > old_max:
            if current_chunk:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            sentences = split_into_sentences(para)
            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_tok = count_tokens(sent, tokenizer)
                if sent_tokens + sent_tok > old_max and sent_chunk:
                    raw_chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_tokens = 0
                sent_chunk.append(sent)
                sent_tokens += sent_tok

            if sent_chunk:
                leftover = " ".join(sent_chunk)
                if sent_tokens >= old_min:
                    raw_chunks.append(leftover)
                else:
                    current_chunk = [leftover]
                    current_tokens = sent_tokens
            continue

        if current_tokens + para_tokens > old_max and current_chunk:
            if current_tokens >= old_min:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if current_tokens < old_min and raw_chunks:
            raw_chunks[-1] = raw_chunks[-1] + "\n\n" + chunk_text
        else:
            raw_chunks.append(chunk_text)

    # Fixed 1-sentence overlap
    final_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0:
            overlap = get_last_sentences(raw_chunks[i-1], 1)
            overlap_tokens = count_tokens(overlap, tokenizer)
            if 10 < overlap_tokens < 100:
                chunk_text = overlap + " " + chunk_text

        chunk = {
            "text": chunk_text,
            "title": metadata["title"],
            "date": metadata["date"],
            "source": metadata["source"],
            "keywords": metadata["keywords"],
            "file": metadata["file"],
            "chunk_index": i,
        }
        final_chunks.append(chunk)

    return final_chunks


def process_all_documents(
    context_mode: str = "simple",
    output_file: Path = None,
    max_docs: int = None
):
    """
    Process all markdown files and output chunks (v3).

    Args:
        context_mode: "simple" (fast, free) or "llm" (slow, costly, better quality)
        output_file: Custom output file path (default: chunks.jsonl for simple, chunks_contextual.jsonl for llm)
        max_docs: Maximum number of documents to process (for testing)
    """
    from contextual_utils import get_anthropic_client, ContextGenerationStats
    import time

    tokenizer = load_tokenizer()

    # Initialize Anthropic client if using LLM mode
    anthropic_client = None
    stats_tracker = None
    if context_mode == "llm":
        anthropic_client = get_anthropic_client()
        if not anthropic_client:
            print("Error: ANTHROPIC_API_KEY not found in environment")
            print("Please set ANTHROPIC_API_KEY to use LLM context mode")
            return
        stats_tracker = ContextGenerationStats()
        print("Using LLM context mode (Claude Haiku with prompt caching)")

    # Determine output file
    if output_file is None:
        output_file = OUTPUT_FILE if context_mode == "simple" else Path(__file__).parent.parent / "chunks_contextual.jsonl"

    # Collect all markdown files
    md_files = []
    for subdir in ["substack", "sevan"]:
        dir_path = FORMATTED_DIR / subdir
        if dir_path.exists():
            md_files.extend(dir_path.glob("*.md"))

    # Limit number of documents if specified
    if max_docs:
        md_files = md_files[:max_docs]

    print(f"Found {len(md_files)} markdown files")
    print(f"Context mode: {context_mode}")
    print(f"Output: {output_file}")
    print()

    all_chunks = []
    skipped_docs = 0
    start_time = time.time()

    for i, file_path in enumerate(md_files):
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(md_files) - i - 1) / rate if rate > 0 else 0

            print(f"Processing {i + 1}/{len(md_files)} ({(i+1)/len(md_files)*100:.1f}%) - "
                  f"Rate: {rate:.1f} docs/sec - ETA: {eta/60:.1f} min")

            # Show cost stats for LLM mode
            if stats_tracker and stats_tracker.num_requests > 0:
                summary = stats_tracker.summary()
                print(f"  Chunks processed: {summary['requests']}")
                print(f"  Cost so far: ${summary['cost_usd']:.4f}")
                print(f"  Cache hit rate: {summary['cache_hit_rate']:.1%}")

        try:
            metadata = parse_markdown(file_path)
            chunks = chunk_document(
                metadata,
                tokenizer,
                enrich_context=True,
                context_mode=context_mode,
                anthropic_client=anthropic_client,
                stats_tracker=stats_tracker
            )
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped_docs += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped_docs += 1

    # Write output
    print(f"\nWriting {len(all_chunks)} chunks to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Print stats
    elapsed_total = time.time() - start_time
    print(f"\nDone!")
    print(f"  Documents: {len(md_files)}")
    print(f"  Skipped (empty): {skipped_docs}")
    print(f"  Chunks: {len(all_chunks)}")
    if len(md_files) > skipped_docs:
        print(f"  Avg chunks/doc: {len(all_chunks) / (len(md_files) - skipped_docs):.1f}")
    print(f"  Total time: {elapsed_total/60:.1f} min")

    # Print LLM stats
    if stats_tracker and stats_tracker.num_requests > 0:
        summary = stats_tracker.summary()
        print(f"\nLLM Context Generation Stats:")
        print(f"  Total requests: {summary['requests']}")
        print(f"  Input tokens: {summary['total_tokens']['input']:,}")
        print(f"  Output tokens: {summary['total_tokens']['output']:,}")
        print(f"  Cache creation: {summary['total_tokens']['cache_creation']:,}")
        print(f"  Cache reads: {summary['total_tokens']['cache_read']:,}")
        print(f"  Cache hit rate: {summary['cache_hit_rate']:.1%}")
        print(f"  Total cost: ${summary['cost_usd']:.4f}")


def process_sample_documents(file_paths: list[Path], compare: bool = True):
    """
    Process a list of sample documents and optionally compare v2 vs v3.

    Args:
        file_paths: List of markdown file paths to process
        compare: If True, output both v2 and v3 for comparison
    """
    tokenizer = load_tokenizer()

    print(f"\n{'='*60}")
    print("SAMPLE DOCUMENT CHUNKING COMPARISON")
    print(f"{'='*60}\n")

    for file_path in file_paths:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        metadata = parse_markdown(file_path)
        print(f"\n{'─'*60}")
        print(f"Document: {metadata['title'] or file_path.name}")
        print(f"Date: {metadata['date']}")
        print(f"Keywords: {metadata['keywords']}")
        print(f"{'─'*60}")

        # Process with v3 (new)
        chunks_v3 = chunk_document(metadata, tokenizer, enrich_context=True)
        print(f"\n[V3 - New] {len(chunks_v3)} chunks")

        if compare:
            # Process with v2 (old)
            chunks_v2 = chunk_document_v2(metadata, tokenizer)
            print(f"[V2 - Old] {len(chunks_v2)} chunks")

        # Show comparison
        print("\n" + "─"*40)
        print("V3 CHUNKS (with context enrichment):")
        print("─"*40)

        for i, chunk in enumerate(chunks_v3[:3]):  # Show first 3 chunks
            tokens = count_tokens(chunk["text_for_embedding"], tokenizer)
            print(f"\n[Chunk {i}] ({tokens} tokens)")
            print(f"  chunk_id: {chunk['chunk_id']}")
            print(f"  prechunk_id: {chunk['prechunk_id']}")
            print(f"  postchunk_id: {chunk['postchunk_id']}")
            print(f"\n  text_for_embedding (first 500 chars):")
            print(f"  {chunk['text_for_embedding'][:500]}...")

        if compare:
            print("\n" + "─"*40)
            print("V2 CHUNKS (old format):")
            print("─"*40)

            for i, chunk in enumerate(chunks_v2[:3]):
                tokens = count_tokens(chunk["text"], tokenizer)
                print(f"\n[Chunk {i}] ({tokens} tokens)")
                print(f"  text (first 500 chars):")
                print(f"  {chunk['text'][:500]}...")

        # Statistics comparison
        print("\n" + "─"*40)
        print("STATISTICS COMPARISON:")
        print("─"*40)

        v3_tokens = [count_tokens(c["text_for_embedding"], tokenizer) for c in chunks_v3]
        print(f"\nV3 (New):")
        print(f"  Total chunks: {len(chunks_v3)}")
        print(f"  Avg tokens/chunk: {sum(v3_tokens)/len(v3_tokens):.0f}")
        print(f"  Min/Max tokens: {min(v3_tokens)}/{max(v3_tokens)}")

        if compare:
            v2_tokens = [count_tokens(c["text"], tokenizer) for c in chunks_v2]
            print(f"\nV2 (Old):")
            print(f"  Total chunks: {len(chunks_v2)}")
            print(f"  Avg tokens/chunk: {sum(v2_tokens)/len(v2_tokens):.0f}")
            print(f"  Min/Max tokens: {min(v2_tokens)}/{max(v2_tokens)}")

    return chunks_v3, chunks_v2 if compare else None


def list_sample_files(n: int = 5) -> list[Path]:
    """List sample markdown files for testing."""
    md_files = []
    for subdir in ["substack", "sevan"]:
        dir_path = FORMATTED_DIR / subdir
        if dir_path.exists():
            md_files.extend(list(dir_path.glob("*.md"))[:n])
    return md_files[:n]


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk documents for RAG with optional LLM-based contextual enrichment"
    )
    parser.add_argument(
        "--context-mode",
        choices=["simple", "llm"],
        default="simple",
        help="Context enrichment mode: 'simple' (metadata prepending, fast, free) or 'llm' (Claude-generated, slow, costly)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: chunks.jsonl for simple, chunks_contextual.jsonl for llm)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample documents for comparison (legacy mode)"
    )

    args = parser.parse_args()

    if args.sample:
        # Legacy sample mode
        sample_files = list_sample_files(3)
        if sample_files:
            process_sample_documents(sample_files, compare=True)
        else:
            print("No sample files found in formatted/ directory")
    else:
        # Full processing with context mode
        output_file = Path(args.output) if args.output else None
        process_all_documents(
            context_mode=args.context_mode,
            output_file=output_file,
            max_docs=args.max_docs
        )
