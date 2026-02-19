"""
Advanced Memory Management Module.

High-level API for memory operations with:
- Semantic search with metadata/tag filtering
- Batch operations for efficiency
- TTL (time-to-live) support
- Tag-based organization
- Import/export functionality
"""
from __future__ import annotations

import logging
import hashlib
import threading
import uuid
from typing import Optional, Any
from dataclasses import dataclass, field

from .embedding import EmbeddingEngine, get_engine, DEFAULT_DIM
from .storage import VectorStorage, MemoryItem

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with content and metadata."""
    key: str
    content: str
    score: float
    metadata: dict[str, Any]
    tags: list[str] = field(default_factory=list)


class MemoryManager:
    """
    Advanced memory management with semantic search.
    
    Features:
    - Semantic similarity search
    - Batch operations
    - TTL support
    - Tag-based organization
    - Metadata filtering
    """
    
    # Embedding cache: content_hash -> embedding, avoids re-encoding identical content
    _EMBED_CACHE_MAXSIZE = 10000
    
    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_model: Optional[str] = None,
        model_key: Optional[str] = None,
        embedding_dim: int = DEFAULT_DIM,
        use_gpu: bool = True,
    ):
        """
        Initialize memory manager.
        
        Args:
            db_path: Path to SQLite database
            embedding_model: Direct HuggingFace model name
            model_key: Preset model key (bge-m3, bge-small-zh, minilm)
            embedding_dim: Embedding dimension
            use_gpu: Whether to use GPU for embedding
        """
        self.embedding_engine = EmbeddingEngine(
            model_name=embedding_model,
            model_key=model_key,
            dimension=embedding_dim,
            use_gpu=use_gpu,
        )
        self.storage = VectorStorage(
            db_path=db_path,
            dimension=embedding_dim,
        )
        self._embed_cache: dict[str, 'np.ndarray'] = {}
    
    def _content_hash(self, content: str) -> str:
        """Fast content hash for embedding cache lookup."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_embedding(self, content: str) -> 'np.ndarray':
        """Get embedding for content, using cache when possible."""
        h = self._content_hash(content)
        if h in self._embed_cache:
            return self._embed_cache[h]
        embedding = self.embedding_engine.encode_single(content)
        # Evict oldest if over capacity (simple FIFO eviction)
        if len(self._embed_cache) >= self._EMBED_CACHE_MAXSIZE:
            oldest_key = next(iter(self._embed_cache))
            del self._embed_cache[oldest_key]
        self._embed_cache[h] = embedding
        return embedding
    
    def _generate_key(self, content: str) -> str:
        """Generate a unique key from content hash + uuid suffix."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        unique_suffix = uuid.uuid4().hex[:4]
        return f"{content_hash}_{unique_suffix}"
    
    def save(
        self,
        content: str,
        key: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Save a memory.
        
        Args:
            content: Text content to save
            key: Optional unique key (auto-generated if not provided)
            metadata: Optional metadata dictionary
            tags: Optional tags for organization
            ttl_seconds: Optional time-to-live in seconds
            
        Returns:
            Key of the saved memory
        """
        if key is None:
            key = self._generate_key(content)
        
        embedding = self._get_embedding(content)
        
        self.storage.save(
            key=key,
            content=content,
            embedding=embedding,
            metadata=metadata,
            tags=tags,
            ttl_seconds=ttl_seconds,
        )
        
        logger.debug(f"Saved memory: {key}")
        return key
    
    def save_batch(
        self,
        items: list[dict],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Save multiple memories efficiently.
        
        Args:
            items: List of dicts with keys: content, key (optional), 
                   metadata (optional), tags (optional), ttl_seconds (optional)
            show_progress: Show progress bar
            
        Returns:
            List of saved keys
        """
        if not items:
            return []
        
        # Generate keys and prepare items
        prepared = []
        for item in items:
            key = item.get("key") or self._generate_key(item["content"])
            prepared.append({
                "key": key,
                "content": item["content"],
                "metadata": item.get("metadata"),
                "tags": item.get("tags"),
                "ttl_seconds": item.get("ttl_seconds"),
            })
        
        # Deduplicate contents for encoding: same content → same embedding
        contents = [item["content"] for item in prepared]
        unique_contents = list(dict.fromkeys(contents))  # preserves order, deduplicates
        
        if len(unique_contents) < len(contents):
            # Some duplicates found — encode unique only, map back
            unique_embeddings = self.embedding_engine.encode_batch(
                unique_contents, show_progress=show_progress
            )
            content_to_idx = {c: i for i, c in enumerate(unique_contents)}
            import numpy as np
            embeddings = np.array([unique_embeddings[content_to_idx[c]] for c in contents])
            
            # Populate cache
            for i, c in enumerate(unique_contents):
                h = self._content_hash(c)
                if h not in self._embed_cache:
                    if len(self._embed_cache) >= self._EMBED_CACHE_MAXSIZE:
                        oldest_key = next(iter(self._embed_cache))
                        del self._embed_cache[oldest_key]
                    self._embed_cache[h] = unique_embeddings[i]
        else:
            embeddings = self.embedding_engine.encode_batch(
                contents, show_progress=show_progress
            )
            # Populate cache
            for i, c in enumerate(contents):
                h = self._content_hash(c)
                if h not in self._embed_cache:
                    if len(self._embed_cache) >= self._EMBED_CACHE_MAXSIZE:
                        oldest_key = next(iter(self._embed_cache))
                        del self._embed_cache[oldest_key]
                    self._embed_cache[h] = embeddings[i]
        
        # Save to storage
        self.storage.save_batch(prepared, embeddings)
        
        keys = [item["key"] for item in prepared]
        logger.info(f"Saved {len(keys)} memories in batch")
        return keys
    
    def save_batch_late_chunking(
        self,
        full_text: str,
        items: list[dict],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Save multiple memories using Late Chunking for superior embeddings.
        
        Instead of encoding each chunk independently, this method encodes the
        entire document through the transformer once and then pools token
        embeddings per chunk boundary. Each chunk's embedding carries
        full-document context, dramatically improving retrieval relevance.
        
        Falls back to standard save_batch if the model does not support
        late chunking (non-long-context model or document too long).
        
        Args:
            full_text: The full document text from which chunks were derived.
            items: List of dicts with keys: content, key, metadata, tags, ttl_seconds.
                   Each item's "content" must be a contiguous substring of full_text.
            show_progress: Show progress bar (used only in fallback path).
            
        Returns:
            List of saved keys
        """
        if not items:
            return []
        
        # Generate keys and prepare items
        prepared = []
        for item in items:
            key = item.get("key") or self._generate_key(item["content"])
            prepared.append({
                "key": key,
                "content": item["content"],
                "metadata": item.get("metadata"),
                "tags": item.get("tags"),
                "ttl_seconds": item.get("ttl_seconds"),
            })
        
        chunk_texts = [item["content"] for item in prepared]
        
        # Use late chunking if supported
        if self.embedding_engine.supports_late_chunking:
            logger.info(f"Using Late Chunking for {len(chunk_texts)} chunks")
            embeddings = self.embedding_engine.encode_late_chunks(
                full_text=full_text,
                chunk_texts=chunk_texts,
                normalize=True,
            )
        else:
            logger.info("Late Chunking not supported by model, using standard encode")
            embeddings = self.embedding_engine.encode_batch(
                chunk_texts, show_progress=show_progress
            )
        
        # Populate embedding cache
        for i, c in enumerate(chunk_texts):
            h = self._content_hash(c)
            if h not in self._embed_cache:
                if len(self._embed_cache) >= self._EMBED_CACHE_MAXSIZE:
                    oldest_key = next(iter(self._embed_cache))
                    del self._embed_cache[oldest_key]
                self._embed_cache[h] = embeddings[i]
        
        # Save to storage
        self.storage.save_batch(prepared, embeddings)
        
        keys = [item["key"] for item in prepared]
        logger.info(f"Saved {len(keys)} memories via Late Chunking")
        return keys
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            metadata_filter: Filter by metadata fields (exact match)
            tags_filter: Filter by tags (any match)
            
        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_engine.encode_single(query)
        
        results = self.storage.search_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            metadata_filter=metadata_filter,
            tags_filter=tags_filter,
        )
        
        return [
            SearchResult(
                key=item.key,
                content=item.content,
                score=score,
                metadata=item.metadata,
                tags=item.tags,
            )
            for item, score in results
        ]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        return self.embedding_engine.similarity(text1, text2)
    
    def get(self, key: str) -> Optional[MemoryItem]:
        """Get a memory by key."""
        return self.storage.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        return self.storage.delete(key)
    
    def delete_expired(self) -> int:
        """Delete all expired memories. Returns count deleted."""
        return self.storage.delete_expired()
    
    def update_metadata(self, key: str, metadata: dict, merge: bool = True) -> bool:
        """Update memory metadata."""
        return self.storage.update_metadata(key, metadata, merge)
    
    def update_tags(self, key: str, tags: list[str], merge: bool = True) -> bool:
        """Update memory tags."""
        return self.storage.update_tags(key, tags, merge)
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        tags_filter: Optional[list[str]] = None,
    ) -> list[MemoryItem]:
        """List all memories with optional tag filtering."""
        return self.storage.list_all(
            limit=limit, 
            offset=offset,
            tags_filter=tags_filter,
        )
    
    def list_by_tags(self, tags: list[str], limit: int = 100) -> list[MemoryItem]:
        """List memories with specific tags."""
        return self.storage.list_by_tags(tags, limit)
    
    def count(self) -> int:
        """Get total number of active memories."""
        return self.storage.count()
    
    def search_text(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
    ) -> list[MemoryItem]:
        """Search memories by full-text search."""
        return self.storage.search_text(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            tags_filter=tags_filter,
        )
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector similarity and BM25 text search.
        
        Uses Reciprocal Rank Fusion (RRF) to combine results from both
        semantic (vector) and keyword (BM25) search, providing +15-30%
        accuracy improvement over vector-only search.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            threshold: Minimum similarity score for vector search
            metadata_filter: Filter by metadata fields (exact match)
            tags_filter: Filter by tags (any match)
            vector_weight: Weight for vector search results (0-1)
            text_weight: Weight for text search results (0-1)
            
        Returns:
            List of SearchResult objects sorted by combined RRF score
        """
        query_embedding = self.embedding_engine.encode_single(query)
        
        results = self.storage.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            metadata_filter=metadata_filter,
            tags_filter=tags_filter,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )
        
        return [
            SearchResult(
                key=item.key,
                content=item.content,
                score=score,
                metadata=item.metadata,
                tags=item.tags,
            )
            for item, score in results
        ]
    
    def get_stats(self) -> dict:
        """Get memory system statistics."""
        storage_stats = self.storage.get_stats()
        engine_info = self.embedding_engine.get_info()
        
        return {
            **storage_stats,
            "embedding_model": engine_info["model_name"],
            "embedding_dimension": engine_info["output_dimension"],
            "device": engine_info["device"],
        }
    
    def export_all(self) -> list[dict]:
        """Export all memories as list of dictionaries."""
        return self.storage.export_all()
    
    def import_batch(self, items: list[dict]) -> int:
        """
        Import memories from exported data.
        
        Args:
            items: List of dicts from export_all()
            
        Returns:
            Number of imported memories
        """
        # Re-encode all content
        contents = [item["content"] for item in items]
        embeddings = self.embedding_engine.encode_batch(contents, show_progress=True)
        
        prepared = []
        for item in items:
            prepared.append({
                "key": item["key"],
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "tags": item.get("tags", []),
            })
        
        self.storage.save_batch(prepared, embeddings)
        return len(items)
    
    def warmup(self):
        """Warmup the embedding model."""
        self.embedding_engine.warmup()
    
    def close(self):
        """Close resources."""
        self.storage.close()


# Global manager instance
_manager: Optional[MemoryManager] = None
_manager_lock = threading.Lock()


def get_manager(
    db_path: Optional[str] = None,
    **kwargs,
) -> MemoryManager:
    """Get or create the global memory manager."""
    global _manager

    if _manager is None:
        with _manager_lock:
            if _manager is None:
                from pathlib import Path
                import os

                if db_path is None:
                    # Default path
                    home = Path.home()
                    db_path = str(home / ".ultra-light-memory" / "memory.db")

                _manager = MemoryManager(db_path=db_path, **kwargs)

    return _manager
