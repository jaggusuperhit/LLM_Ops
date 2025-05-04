"""
Embedding utilities for LLMOps projects.

This module provides functions to create and manipulate text embeddings.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import json
import pickle

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Load a sentence transformer model.
    
    Args:
        model_name: Name of the model to load.
        
    Returns:
        A SentenceTransformer model.
        
    Raises:
        ImportError: If sentence_transformers is not installed.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence_transformers is not installed. "
            "Please install it with `pip install sentence-transformers`."
        )
    
    return SentenceTransformer(model_name)


def get_embeddings_sentence_transformer(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress_bar: bool = True,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Get embeddings using SentenceTransformer.
    
    Args:
        texts: List of texts to embed.
        model_name: Name of the model to use.
        batch_size: Batch size for embedding.
        show_progress_bar: Whether to show a progress bar.
        normalize_embeddings: Whether to normalize the embeddings.
        
    Returns:
        A numpy array of embeddings.
        
    Raises:
        ImportError: If sentence_transformers is not installed.
    """
    model = load_sentence_transformer(model_name)
    
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize_embeddings,
    )


def get_embeddings_openai(
    texts: List[str],
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    Get embeddings using OpenAI's API.
    
    Args:
        texts: List of texts to embed.
        model: Name of the OpenAI embedding model to use.
        api_key: OpenAI API key. If None, it will use the OPENAI_API_KEY environment variable.
        
    Returns:
        A numpy array of embeddings.
        
    Raises:
        ImportError: If openai is not installed.
        ValueError: If the API key is not provided and not found in environment variables.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai is not installed. "
            "Please install it with `pip install openai`."
        )
    
    if api_key:
        openai.api_key = api_key
    elif not openai.api_key:
        raise ValueError(
            "OpenAI API key not provided. "
            "Please provide an API key or set the OPENAI_API_KEY environment variable."
        )
    
    # Process in batches to avoid rate limits
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [item["embedding"] for item in response["data"]]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between batches of vectors.
    
    Args:
        a: First batch of vectors (n x d).
        b: Second batch of vectors (m x d).
        
    Returns:
        Matrix of cosine similarities (n x m).
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    
    return np.dot(a_normalized, b_normalized.T)


def save_embeddings(
    embeddings: np.ndarray,
    texts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    file_path: str = "embeddings.pkl",
) -> None:
    """
    Save embeddings, texts, and metadata to a file.
    
    Args:
        embeddings: Numpy array of embeddings.
        texts: List of texts corresponding to the embeddings.
        metadata: Optional list of metadata dictionaries.
        file_path: Path to save the embeddings.
    """
    data = {
        "embeddings": embeddings,
        "texts": texts,
    }
    
    if metadata:
        data["metadata"] = metadata
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pkl" or suffix == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif suffix == ".npz":
        if metadata:
            # Convert metadata to a format that can be saved with numpy
            metadata_str = [json.dumps(m) for m in metadata]
            np.savez(path, embeddings=embeddings, texts=np.array(texts, dtype=object), metadata=np.array(metadata_str, dtype=object))
        else:
            np.savez(path, embeddings=embeddings, texts=np.array(texts, dtype=object))
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl, .pickle, or .npz.")


def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings, texts, and metadata from a file.
    
    Args:
        file_path: Path to the embeddings file.
        
    Returns:
        Dictionary containing embeddings, texts, and metadata (if available).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pkl" or suffix == ".pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        result = {
            "embeddings": data["embeddings"],
            "texts": data["texts"].tolist(),
        }
        
        if "metadata" in data:
            # Convert metadata back to dictionaries
            result["metadata"] = [json.loads(m) for m in data["metadata"]]
        
        return result
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl, .pickle, or .npz.")
