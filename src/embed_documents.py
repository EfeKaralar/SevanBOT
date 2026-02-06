#!/usr/bin/env python3
"""
embed_documents.py - Generate embeddings for chunked documents

Usage:
    python3 src/embed_documents.py --model turkembed
    python3 src/embed_documents.py --model bge-m3
    python3 src/embed_documents.py --model turkembed --skip-qdrant

This script:
1. Reads all chunks from chunks_contextual.jsonl (always re-embeds everything)
2. Generates embeddings using the specified model
3. Saves results to embeddings/{model}/ directory
4. Uploads to Qdrant vector database (unless --skip-qdrant)
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from model_registry import (
    get_model_config,
    load_embedding_model,
    get_embedding_params,
    MODEL_CONFIGS
)
from qdrant_helpers import (
    create_qdrant_collection,
    upload_embeddings_to_qdrant,
    get_qdrant_client
)

# Paths
CHUNKS_FILE = Path(__file__).parent.parent / "chunks_contextual.jsonl"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"


def load_chunks(chunks_file: Path) -> List[Dict[str, Any]]:
    """
    Load all chunks from JSONL file.

    Args:
        chunks_file: Path to chunks JSONL file

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
    return chunks


def get_already_embedded_chunk_ids(output_file: Path) -> Set[str]:
    """
    Get set of chunk IDs that are already embedded.

    Args:
        output_file: Path to existing embeddings.jsonl file

    Returns:
        Set of chunk_ids that are already embedded
    """
    if not output_file.exists():
        return set()

    embedded_ids = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                embedded_ids.add(record['chunk_id'])
    except Exception as e:
        print(f"[WARN] Could not read existing embeddings: {e}")
        return set()

    return embedded_ids


def generate_embeddings_openai(
    chunks: List[Dict[str, Any]],
    model_key: str
) -> np.ndarray:
    """
    Generate embeddings using OpenAI API.

    Args:
        chunks: List of chunk dictionaries
        model_key: Model identifier (for config)

    Returns:
        numpy array of embeddings (n_chunks x dimensions)
    """
    from openai import OpenAI

    config = get_model_config(model_key)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    texts_to_embed = [chunk['text_for_embedding'] for chunk in chunks]
    batch_size = config['batch_size']

    print(f"[EMBED] Generating embeddings for {len(texts_to_embed)} chunks...")
    print(f"[EMBED] Using OpenAI API: {config['name']}")
    print(f"[EMBED] Batch size: {batch_size}")

    all_embeddings = []
    start_time = time.time()

    # Process in batches
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        print(f"[EMBED] Processing batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1} ({len(batch)} chunks)")

        response = client.embeddings.create(
            input=batch,
            model=config['name']
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    elapsed = time.time() - start_time
    embeddings = np.array(all_embeddings)

    print(f"[EMBED] Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"[EMBED] Speed: {len(embeddings)/elapsed:.1f} chunks/sec")

    # Calculate cost (rough estimate: 392 tokens/chunk average)
    est_tokens = len(texts_to_embed) * 392
    if 'small' in config['name']:
        cost = est_tokens * 0.020 / 1_000_000
    else:  # large
        cost = est_tokens * 0.130 / 1_000_000
    print(f"[EMBED] Cost estimate: ${cost:.5f} ({config['name']})")

    return embeddings


def generate_embeddings_local(
    chunks: List[Dict[str, Any]],
    model,
    model_key: str
) -> np.ndarray:
    """
    Generate embeddings using local SentenceTransformer model.

    Args:
        chunks: List of chunk dictionaries
        model: Loaded SentenceTransformer model
        model_key: Model identifier (for params)

    Returns:
        numpy array of embeddings (n_chunks x dimensions)
    """
    texts_to_embed = [chunk['text_for_embedding'] for chunk in chunks]
    params = get_embedding_params(model_key)

    print(f"[EMBED] Generating embeddings for {len(texts_to_embed)} chunks...")
    print(f"[EMBED] Batch size: {params['batch_size']}")

    start_time = time.time()
    embeddings = model.encode(texts_to_embed, **params)
    elapsed = time.time() - start_time

    print(f"[EMBED] Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"[EMBED] Speed: {len(embeddings)/elapsed:.1f} chunks/sec")

    return embeddings


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model,
    model_key: str
) -> np.ndarray:
    """
    Generate embeddings for all chunks (routes to API or local).

    Args:
        chunks: List of chunk dictionaries
        model: Loaded SentenceTransformer model (None for API models)
        model_key: Model identifier (for params)

    Returns:
        numpy array of embeddings (n_chunks x dimensions)
    """
    config = get_model_config(model_key)

    if config.get('type') == 'api':
        return generate_embeddings_openai(chunks, model_key)
    else:
        return generate_embeddings_local(chunks, model, model_key)


def save_embeddings_jsonl(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    output_file: Path,
    append: bool = False
) -> None:
    """
    Save embeddings and metadata to JSONL.

    Args:
        chunks: List of chunk dictionaries
        embeddings: numpy array of embeddings
        output_file: Path to output JSONL file
        append: If True, append to existing file. If False, overwrite.
    """
    mode = 'a' if append else 'w'
    action = "Appending" if append else "Writing"
    print(f"[SAVE] {action} {len(embeddings)} embeddings to {output_file}")

    with open(output_file, mode, encoding='utf-8') as f:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'embedding': embedding.tolist(),
                'metadata': {
                    'title': chunk.get('title'),
                    'date': chunk.get('date'),
                    'source': chunk.get('source'),
                    'chunk_index': chunk.get('chunk_index'),
                    'total_chunks': chunk.get('total_chunks'),
                    'context_mode': chunk.get('context_mode'),
                    'llm_context': chunk.get('llm_context')
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"[SAVE] Saved {len(embeddings)} embeddings")


def save_metadata(
    model_key: str,
    num_chunks: int,
    output_dir: Path,
    elapsed_time: float,
    input_file: Path
) -> None:
    """
    Save embedding run metadata.

    Args:
        model_key: Model identifier
        num_chunks: Number of chunks embedded
        output_dir: Output directory
        elapsed_time: Time taken in seconds
        input_file: Input chunks file path
    """
    config = get_model_config(model_key)
    metadata_file = output_dir / "metadata.json"

    metadata = {
        'model_key': model_key,
        'model_name': config['name'],
        'dimensions': config['dimensions'],
        'num_chunks': num_chunks,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed_time, 2),
        'chunks_per_second': round(num_chunks / elapsed_time, 2),
        'input_file': str(input_file)
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for chunked documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/embed_documents.py --model turkembed
  python3 src/embed_documents.py --model bge-m3
  python3 src/embed_documents.py --model turkembed --skip-qdrant

Available models: """ + ", ".join(MODEL_CONFIGS.keys())
    )

    parser.add_argument(
        '--model',
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help='Embedding model to use'
    )
    parser.add_argument(
        '--chunks-file',
        type=Path,
        default=CHUNKS_FILE,
        help=f'Path to chunks JSONL file (default: {CHUNKS_FILE})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: embeddings/{model}/)'
    )
    parser.add_argument(
        '--skip-qdrant',
        action='store_true',
        help='Skip uploading to Qdrant (only save JSONL)'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only embed new chunks (skip already-embedded ones)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-embed all chunks (ignore existing embeddings)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = EMBEDDINGS_DIR / args.model

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"EMBEDDING GENERATION - Model: {args.model.upper()}")
    print("="*70)

    # Load chunks
    print(f"\n[LOAD] Reading chunks from {args.chunks_file}")
    all_chunks = load_chunks(args.chunks_file)
    print(f"[LOAD] Loaded {len(all_chunks)} chunks")

    # Check for existing embeddings (incremental mode)
    output_file = args.output_dir / "embeddings.jsonl"
    if args.incremental and not args.force and output_file.exists():
        print(f"\n[INCREMENTAL] Checking for already-embedded chunks...")
        already_embedded = get_already_embedded_chunk_ids(output_file)
        print(f"[INCREMENTAL] Found {len(already_embedded)} already-embedded chunks")

        # Filter to only new chunks
        chunks = [c for c in all_chunks if c['chunk_id'] not in already_embedded]
        print(f"[INCREMENTAL] Will embed {len(chunks)} new chunks")

        if len(chunks) == 0:
            print(f"[INCREMENTAL] No new chunks to embed. Exiting.")
            return
    else:
        chunks = all_chunks
        if args.force:
            print(f"\n[FORCE] Re-embedding all chunks (ignoring existing embeddings)")

    # Load model
    print(f"\n[MODEL] Initializing {args.model}")
    model = load_embedding_model(args.model)
    config = get_model_config(args.model)

    # Generate embeddings
    print(f"\n[EMBED] Starting embedding generation...")
    start_time = time.time()
    embeddings = generate_embeddings(chunks, model, args.model)
    elapsed = time.time() - start_time

    # Save to JSONL
    print(f"\n[SAVE] Saving results to {args.output_dir}")
    output_file = args.output_dir / "embeddings.jsonl"
    append_mode = args.incremental and not args.force and output_file.exists()
    save_embeddings_jsonl(chunks, embeddings, output_file, append=append_mode)
    save_metadata(args.model, len(chunks), args.output_dir, elapsed, args.chunks_file)

    # Upload to Qdrant
    if not args.skip_qdrant:
        print(f"\n[QDRANT] Uploading to Qdrant...")
        try:
            client = get_qdrant_client()
            collection_name = f"sevanbot_{args.model}"

            create_qdrant_collection(
                client,
                collection_name,
                vector_size=config['dimensions']
            )

            upload_embeddings_to_qdrant(
                client,
                collection_name,
                chunks,
                embeddings
            )

            print(f"[QDRANT] Upload complete!")
        except Exception as e:
            print(f"[ERROR] Qdrant upload failed: {e}")
            print(f"[ERROR] Embeddings saved to JSONL, you can retry upload later")

    # Final summary
    print("\n" + "="*70)
    print("EMBEDDING COMPLETE!")
    print("="*70)
    print(f"Model:       {config['name']}")
    print(f"Chunks:      {len(chunks)}")
    print(f"Dimensions:  {config['dimensions']}")
    print(f"Time:        {elapsed:.1f}s")
    print(f"Speed:       {len(chunks)/elapsed:.1f} chunks/sec")
    print(f"Output:      {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
