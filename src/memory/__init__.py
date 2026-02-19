"""
Ultra-Light Memory MCP Server.

A lightweight AI memory system with semantic search capabilities.

Features:
- Semantic search with BAAI/bge embeddings
- SIMD-accelerated vector storage via sqlite-vec
- Batch operations, TTL, and tag-based organization
- Thread-safe with connection pooling
"""

__version__ = "0.2.0"

from .memory import MemoryManager, SearchResult, get_manager
from .embedding import EmbeddingEngine, get_engine, list_models, DEFAULT_DIM
from .storage import VectorStorage, MemoryItem

__all__ = [
    # Version
    "__version__",
    # Memory Manager
    "MemoryManager",
    "SearchResult", 
    "get_manager",
    # Embedding
    "EmbeddingEngine",
    "get_engine",
    "list_models",
    "DEFAULT_DIM",
    # Storage
    "VectorStorage",
    "MemoryItem",
]