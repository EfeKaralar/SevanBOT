"""
Model registry for embedding models.
Maps model names to their configurations and loading functions.
Supports both local models (SentenceTransformer) and API-based models (OpenAI).
"""

from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Model configuration registry (simple dict)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "turkembed": {
        "name": "newmindai/TurkEmbed4Retrieval",
        "dimensions": 768,
        "max_seq_length": 512,
        "batch_size": 8,  # Reduced for memory-constrained systems
        "normalize_embeddings": True,
        "description": "Turkish-optimized embedding model"
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimensions": 1024,
        "max_seq_length": 8192,
        "batch_size": 16,
        "normalize_embeddings": True,
        "description": "Multilingual BGE model with long context"
    },
    "openai-small": {
        "name": "text-embedding-3-small",
        "type": "api",
        "provider": "openai",
        "dimensions": 1536,
        "max_seq_length": 8191,
        "batch_size": 100,  # OpenAI allows large batches
        "description": "OpenAI API - fast, cheap, zero local RAM"
    },
    "openai-large": {
        "name": "text-embedding-3-large",
        "type": "api",
        "provider": "openai",
        "dimensions": 3072,
        "max_seq_length": 8191,
        "batch_size": 100,
        "description": "OpenAI API - highest quality, still cheap"
    }
}


def get_model_config(model_key: str) -> Dict[str, Any]:
    """
    Get configuration for a model by key.

    Args:
        model_key: Model identifier (e.g., 'turkembed', 'bge-m3')

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model_key is unknown
    """
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return MODEL_CONFIGS[model_key]


def load_embedding_model(model_key: str) -> Optional[SentenceTransformer]:
    """
    Load embedding model by key.

    Args:
        model_key: Model identifier (e.g., 'turkembed', 'bge-m3')

    Returns:
        Loaded SentenceTransformer model

    Raises:
        ValueError: If model_key is unknown
    """
    import torch

    config = get_model_config(model_key)
    print(f"[LOAD] Loading model: {config['name']}")
    print(f"[LOAD] Dimensions: {config['dimensions']}, Max length: {config['max_seq_length']}")

    # API-based models don't need local loading
    if config.get('type') == 'api':
        print(f"[LOAD] API-based model ({config['provider'].upper()}) - no local model needed")
        return None

    # Detect best device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"[LOAD] Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"[LOAD] Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print(f"[LOAD] Using CPU (slow)")

    model = SentenceTransformer(config['name'], trust_remote_code=True, device=device)

    # Configure model
    model.max_seq_length = config['max_seq_length']

    print(f"[LOAD] Model loaded successfully on {device}")
    return model


def get_embedding_params(model_key: str) -> Dict[str, Any]:
    """
    Get encoding parameters for a model.

    Args:
        model_key: Model identifier (e.g., 'turkembed', 'bge-m3')

    Returns:
        Dictionary of parameters for model.encode()

    Raises:
        ValueError: If model_key is unknown
    """
    config = get_model_config(model_key)
    return {
        'batch_size': config['batch_size'],
        'normalize_embeddings': config['normalize_embeddings'],
        'show_progress_bar': True,
        'convert_to_numpy': True
    }
