"""
Qdrant helper functions for vector storage.
"""

from typing import List, Dict, Any
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)

QDRANT_PATH = os.getenv("QDRANT_PATH", ".qdrant")  # Local storage
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_qdrant_client(host: str = None) -> QdrantClient:
    """
    Get Qdrant client (local or remote).

    Args:
        host: Remote host URL (None for local storage)

    Returns:
        QdrantClient instance
    """
    host = host or QDRANT_URL
    if host:
        print(f"[QDRANT] Connecting to {host}")
        return QdrantClient(url=host, api_key=QDRANT_API_KEY)
    else:
        print(f"[QDRANT] Using local storage: {QDRANT_PATH}")
        return QdrantClient(path=QDRANT_PATH)


def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE
) -> None:
    """
    Create or recreate a Qdrant collection.

    Args:
        client: QdrantClient instance
        collection_name: Name of collection
        vector_size: Dimension of embeddings
        distance: Distance metric (COSINE, DOT, EUCLID)
    """
    # Always recreate (user requirement: always re-embed everything)
    if client.collection_exists(collection_name):
        print(f"[QDRANT] Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    print(f"[QDRANT] Creating collection: {collection_name}")
    print(f"[QDRANT] Vector size: {vector_size}, Distance: {distance.value}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )


def upload_embeddings_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    batch_size: int = 100
) -> None:
    """
    Upload embeddings to Qdrant in batches.

    Args:
        client: QdrantClient instance
        collection_name: Target collection
        chunks: List of chunk dictionaries
        embeddings: numpy array of embeddings
        batch_size: Upload batch size
    """
    total = len(chunks)
    print(f"[QDRANT] Uploading {total} points to {collection_name}")
    print(f"[QDRANT] Batch size: {batch_size}")

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        points = []
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            point = PointStruct(
                id=hash(chunk['chunk_id']) % (2**63 - 1),  # Stable hash
                vector=embedding.tolist(),
                payload={
                    'chunk_id': chunk['chunk_id'],
                    'doc_id': chunk['doc_id'],
                    'text': chunk['text'],
                    'title': chunk.get('title'),
                    'date': chunk.get('date'),
                    'source': chunk.get('source'),
                    'chunk_index': chunk.get('chunk_index'),
                    'total_chunks': chunk.get('total_chunks'),
                    # Include new contextual fields
                    'context_mode': chunk.get('context_mode'),
                    'llm_context': chunk.get('llm_context')
                }
            )
            points.append(point)

        client.upsert(collection_name=collection_name, points=points)

        progress = min(i + batch_size, total)
        print(f"[QDRANT] Uploaded {progress}/{total} points")

    print(f"[QDRANT] Upload complete!")


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Qdrant collection.

    Args:
        client: QdrantClient instance
        collection_name: Collection to search
        query_vector: Query embedding
        limit: Number of results

    Returns:
        List of search results with scores
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )

    return [
        {
            'chunk_id': hit.payload['chunk_id'],
            'score': hit.score,
            'text': hit.payload['text'],
            'metadata': {
                'title': hit.payload.get('title'),
                'date': hit.payload.get('date'),
                'source': hit.payload.get('source'),
                'context_mode': hit.payload.get('context_mode'),
                'llm_context': hit.payload.get('llm_context')
            }
        }
        for hit in results
    ]
