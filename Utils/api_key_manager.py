"""
API Key Manager for LLMOps projects.

This utility helps manage API keys securely for various services used in LLMOps projects.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIKeyManager:
    """
    A class to manage API keys for various services.
    
    This class provides methods to:
    - Get API keys from environment variables
    - Load API keys from a JSON file
    - Save API keys to a JSON file (not recommended for production)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the APIKeyManager.
        
        Args:
            config_path: Optional path to a JSON config file containing API keys.
        """
        self.keys = {}
        
        # Try to load from environment variables first
        self._load_from_env()
        
        # If config_path is provided, try to load from there as well
        if config_path:
            self._load_from_file(config_path)
    
    def _load_from_env(self) -> None:
        """Load API keys from environment variables."""
        # Common API keys used in LLMOps projects
        env_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
            "pinecone": os.getenv("PINECONE_API_KEY"),
            "weaviate": os.getenv("WEAVIATE_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "azure_openai": os.getenv("AZURE_OPENAI_API_KEY"),
            "langchain": os.getenv("LANGCHAIN_API_KEY"),
            "langsmith": os.getenv("LANGSMITH_API_KEY"),
        }
        
        # Only add keys that are not None
        self.keys.update({k: v for k, v in env_keys.items() if v is not None})
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load API keys from a JSON file.
        
        Args:
            config_path: Path to a JSON config file containing API keys.
        """
        path = Path(config_path)
        if path.exists() and path.is_file():
            try:
                with open(path, "r") as f:
                    file_keys = json.load(f)
                    # Only add keys that are not None or empty
                    self.keys.update({k: v for k, v in file_keys.items() if v})
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config file: {e}")
    
    def get_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service.
        
        Args:
            service: The name of the service (e.g., 'openai', 'huggingface').
            
        Returns:
            The API key if found, None otherwise.
        """
        return self.keys.get(service.lower())
    
    def set_key(self, service: str, key: str) -> None:
        """
        Set an API key for a specific service.
        
        Args:
            service: The name of the service (e.g., 'openai', 'huggingface').
            key: The API key to set.
        """
        self.keys[service.lower()] = key
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save API keys to a JSON file (not recommended for production).
        
        Args:
            config_path: Path to save the JSON config file.
        """
        try:
            with open(config_path, "w") as f:
                json.dump(self.keys, f, indent=2)
        except IOError as e:
            print(f"Error saving config file: {e}")
    
    def get_all_keys(self) -> Dict[str, str]:
        """
        Get all available API keys.
        
        Returns:
            A dictionary of all available API keys.
        """
        return self.keys.copy()


# Example usage
if __name__ == "__main__":
    # Create an API key manager
    key_manager = APIKeyManager()
    
    # Get an API key
    openai_key = key_manager.get_key("openai")
    if openai_key:
        print("OpenAI API key found!")
    else:
        print("OpenAI API key not found.")
    
    # Print all available keys (masked for security)
    all_keys = key_manager.get_all_keys()
    for service, key in all_keys.items():
        if key:
            masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:]
            print(f"{service}: {masked_key}")
