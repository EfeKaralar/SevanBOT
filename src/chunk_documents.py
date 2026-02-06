#!/usr/bin/env python3
"""
Semantic chunking for SevanBot documents (v2).

Splits markdown articles into chunks suitable for embedding:
- Primary split: paragraphs
- Merges small paragraphs until minimum size reached
- Fallback: sentences (if paragraph > max tokens)
- Sentence-level overlap between chunks
- Filters noise (footnotes, images, Ottoman examples)
"""

import json
import re
from pathlib import Path
from transformers import AutoTokenizer

# Directories
FORMATTED_DIR = Path(__file__).parent.parent / "formatted"
OUTPUT_FILE = Path(__file__).parent.parent / "chunks.jsonl"

# Chunking parameters
MAX_TOKENS = 400
MIN_TOKENS = 150
OVERLAP_SENTENCES = 1

# Ottoman text detection (special diacritics used in transliteration)
OTTOMAN_PATTERN = re.compile(r'[ˁāīūḥḍṣṭẓġʿʾ]')


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


def get_last_sentences(text: str, n: int = 1) -> str:
    """Get last n sentences from text for overlap."""
    sentences = split_into_sentences(text)
    if len(sentences) <= n:
        return text
    return " ".join(sentences[-n:])


def chunk_document(metadata: dict, tokenizer) -> list[dict]:
    """
    Chunk a document into pieces suitable for embedding.
    Returns list of chunk dicts with text and metadata.
    """
    content = metadata["content"]
    if not content:
        return []

    # Extract paragraphs
    paragraphs = extract_paragraphs(content)

    # Filter noise
    paragraphs = [p for p in paragraphs if not is_noise(p) and not is_ottoman_example(p)]

    if not paragraphs:
        return []

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
            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_tok = count_tokens(sent, tokenizer)
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

    # Add sentence-level overlap
    final_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and OVERLAP_SENTENCES > 0:
            overlap = get_last_sentences(raw_chunks[i-1], OVERLAP_SENTENCES)
            # Only add if overlap is meaningful and not too long
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


def process_all_documents():
    """Process all markdown files and output chunks."""
    tokenizer = load_tokenizer()

    # Collect all markdown files
    md_files = []
    for subdir in ["substack", "sevan"]:
        dir_path = FORMATTED_DIR / subdir
        if dir_path.exists():
            md_files.extend(dir_path.glob("*.md"))

    print(f"Found {len(md_files)} markdown files")

    all_chunks = []
    skipped_docs = 0
    for i, file_path in enumerate(md_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(md_files)}...")

        try:
            metadata = parse_markdown(file_path)
            chunks = chunk_document(metadata, tokenizer)
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped_docs += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Write output
    print(f"\nWriting {len(all_chunks)} chunks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\nDone!")
    print(f"  Documents: {len(md_files)}")
    print(f"  Skipped (empty): {skipped_docs}")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Avg chunks/doc: {len(all_chunks) / (len(md_files) - skipped_docs):.1f}")


if __name__ == "__main__":
    process_all_documents()
