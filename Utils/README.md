# Utility Scripts for LLMOps

This directory contains utility scripts and tools to help with LLMOps tasks.

## Available Utilities

### API Key Manager (`api_key_manager.py`)

A utility for managing API keys securely for various services used in LLMOps projects.

**Features:**
- Load API keys from environment variables
- Load API keys from a JSON configuration file
- Save API keys to a JSON file (not recommended for production)
- Get and set API keys for specific services

**Example usage:**
```python
from Utils.api_key_manager import APIKeyManager

# Create an API key manager
key_manager = APIKeyManager()

# Get an API key
openai_key = key_manager.get_key("openai")
if openai_key:
    print("OpenAI API key found!")
else:
    print("OpenAI API key not found.")
```

### Embedding Utilities (`embedding_utils.py`)

A collection of functions for creating and manipulating text embeddings.

**Features:**
- Generate embeddings using SentenceTransformers
- Generate embeddings using OpenAI's API
- Calculate cosine similarity between vectors
- Save and load embeddings to/from files

**Example usage:**
```python
from Utils.embedding_utils import get_embeddings_sentence_transformer, cosine_similarity

# Get embeddings for a list of texts
texts = ["Hello, world!", "How are you?", "I'm fine, thank you."]
embeddings = get_embeddings_sentence_transformer(texts)

# Calculate cosine similarity between two embeddings
similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

## Adding New Utilities

When adding new utility scripts, please follow these guidelines:

1. Create a new Python file with a descriptive name
2. Add comprehensive docstrings and type hints
3. Include example usage in the script
4. Update this README.md file with information about the new utility
