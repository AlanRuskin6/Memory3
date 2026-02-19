"""
Comprehensive test suite for Ultra-Light Memory.

Tests cover:
- Embedding engine
- Vector storage
- Memory manager
- MCP server tools
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


# ============================================================================
# Embedding Tests
# ============================================================================

class TestEmbedding:
    """Tests for the embedding module."""
    
    def test_engine_init(self):
        """Test engine initialization with default settings."""
        from memory.embedding import EmbeddingEngine, DEFAULT_DIM
        
        engine = EmbeddingEngine(use_gpu=False)
        assert engine.dimension == DEFAULT_DIM
        assert engine.model_key in ["bge-m3", "bge-small-zh", "minilm", "custom"]
    
    def test_encode_single(self):
        """Test encoding a single text."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        embedding = engine.encode_single("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (engine.dimension,)
        assert embedding.dtype == np.float32
    
    def test_encode_batch(self):
        """Test encoding multiple texts."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        texts = ["Hello", "World", "Test"]
        embeddings = engine.encode(texts)
        
        assert embeddings.shape == (3, engine.dimension)
    
    def test_encode_empty(self):
        """Test encoding empty list."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        embeddings = engine.encode([])
        
        assert embeddings.shape == (0, engine.dimension)
    
    def test_similarity(self):
        """Test similarity calculation."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        
        # Similar texts should have high similarity
        sim1 = engine.similarity("The cat sat on the mat", "A cat is sitting on a mat")
        assert sim1 > 0.5
        
        # Different texts should have lower similarity
        sim2 = engine.similarity("The cat sat on the mat", "Python programming language")
        assert sim2 < sim1
    
    def test_matryoshka_dimension(self):
        """Test dimension truncation."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(dimension=128, use_gpu=False)
        embedding = engine.encode_single("Test")
        
        assert embedding.shape == (128,)
    
    def test_normalization(self):
        """Test that embeddings are normalized."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        embedding = engine.encode_single("Test", normalize=True)
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    def test_list_models(self):
        """Test model listing."""
        from memory.embedding import list_models
        
        models = list_models()
        assert "bge-small-zh" in models
        assert "minilm" in models
        assert "bge-m3" in models
    
    def test_get_info(self):
        """Test engine info."""
        from memory.embedding import EmbeddingEngine
        
        engine = EmbeddingEngine(use_gpu=False)
        engine.encode("warmup")  # Force load
        
        info = engine.get_info()
        assert "model_name" in info
        assert "output_dimension" in info
        assert "device" in info


# ============================================================================
# Storage Tests
# ============================================================================

class TestStorage:
    """Tests for the vector storage module."""
    
    def test_storage_init(self):
        """Test storage initialization."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        assert storage.dimension == 384
    
    def test_save_and_get(self):
        """Test saving and retrieving a memory."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(
            key="test1",
            content="Test content",
            embedding=embedding,
            metadata={"source": "test"},
            tags=["tag1", "tag2"],
        )
        
        item = storage.get("test1")
        assert item is not None
        assert item.key == "test1"
        assert item.content == "Test content"
        assert item.metadata == {"source": "test"}
        assert item.tags == ["tag1", "tag2"]
    
    def test_save_with_ttl(self):
        """Test saving with TTL."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        # Save with very short TTL (will be expired by the time we check)
        storage.save(
            key="expiring",
            content="Will expire",
            embedding=embedding,
            ttl_seconds=3600,  # 1 hour
        )
        
        item = storage.get("expiring")
        assert item is not None
        assert item.expires_at is not None
    
    def test_delete(self):
        """Test deleting a memory."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(key="to_delete", content="Delete me", embedding=embedding)
        assert storage.get("to_delete") is not None
        
        result = storage.delete("to_delete")
        assert result is True
        assert storage.get("to_delete") is None
    
    def test_save_batch(self):
        """Test batch saving."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        
        items = [
            {"key": "batch1", "content": "Content 1"},
            {"key": "batch2", "content": "Content 2"},
            {"key": "batch3", "content": "Content 3"},
        ]
        embeddings = np.random.randn(3, 384).astype(np.float32)
        
        ids = storage.save_batch(items, embeddings)
        assert len(ids) == 3
        
        assert storage.get("batch1") is not None
        assert storage.get("batch2") is not None
        assert storage.get("batch3") is not None
    
    def test_vector_search(self):
        """Test vector similarity search."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        
        # Create embeddings with known similarities
        base = np.random.randn(384).astype(np.float32)
        base = base / np.linalg.norm(base)
        
        similar = base + np.random.randn(384).astype(np.float32) * 0.1
        similar = similar / np.linalg.norm(similar)
        
        different = np.random.randn(384).astype(np.float32)
        different = different / np.linalg.norm(different)
        
        storage.save(key="similar", content="Similar", embedding=similar)
        storage.save(key="different", content="Different", embedding=different)
        
        results = storage.search_vector(base, top_k=10, threshold=0.0)
        assert len(results) >= 1
        
        # Similar should rank higher
        keys = [item.key for item, _ in results]
        if "similar" in keys and "different" in keys:
            assert keys.index("similar") < keys.index("different")
    
    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        storage.save(key="m1", content="Memory 1", embedding=embedding, 
                     metadata={"type": "note"})
        storage.save(key="m2", content="Memory 2", embedding=embedding,
                     metadata={"type": "task"})
        
        results = storage.search_vector(
            embedding, top_k=10,
            metadata_filter={"type": "note"}
        )
        
        keys = [item.key for item, _ in results]
        assert "m1" in keys
        assert "m2" not in keys
    
    def test_search_with_tags_filter(self):
        """Test search with tag filtering."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        storage.save(key="t1", content="Tagged 1", embedding=embedding,
                     tags=["work", "important"])
        storage.save(key="t2", content="Tagged 2", embedding=embedding,
                     tags=["personal"])
        
        results = storage.search_vector(
            embedding, top_k=10,
            tags_filter=["work"]
        )
        
        keys = [item.key for item, _ in results]
        assert "t1" in keys
        assert "t2" not in keys
    
    def test_list_all(self):
        """Test listing all memories."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        for i in range(5):
            storage.save(key=f"list{i}", content=f"Content {i}", embedding=embedding)
        
        items = storage.list_all(limit=3)
        assert len(items) == 3
        
        items = storage.list_all(limit=10)
        assert len(items) == 5
    
    def test_count(self):
        """Test counting memories."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        assert storage.count() == 0
        
        storage.save(key="c1", content="Content 1", embedding=embedding)
        assert storage.count() == 1
        
        storage.save(key="c2", content="Content 2", embedding=embedding)
        assert storage.count() == 2
    
    def test_update_metadata(self):
        """Test updating metadata."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(key="update", content="Content", embedding=embedding,
                     metadata={"a": 1})
        
        storage.update_metadata("update", {"b": 2}, merge=True)
        item = storage.get("update")
        assert item.metadata == {"a": 1, "b": 2}
        
        storage.update_metadata("update", {"c": 3}, merge=False)
        item = storage.get("update")
        assert item.metadata == {"c": 3}
    
    def test_update_tags(self):
        """Test updating tags."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(key="tags", content="Content", embedding=embedding,
                     tags=["a", "b"])
        
        storage.update_tags("tags", ["c"], merge=True)
        item = storage.get("tags")
        assert set(item.tags) == {"a", "b", "c"}
    
    def test_get_stats(self):
        """Test getting statistics."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(key="s1", content="Content", embedding=embedding)
        
        stats = storage.get_stats()
        assert stats["total_memories"] == 1
        assert stats["with_embedding"] == 1
        assert stats["dimension"] == 384
    
    def test_export_all(self):
        """Test exporting all memories."""
        from memory.storage import VectorStorage
        
        storage = VectorStorage(db_path=":memory:")
        embedding = np.random.randn(384).astype(np.float32)
        
        storage.save(key="e1", content="Export 1", embedding=embedding,
                     metadata={"x": 1}, tags=["tag"])
        
        exported = storage.export_all()
        assert len(exported) == 1
        assert exported[0]["key"] == "e1"
        assert exported[0]["content"] == "Export 1"
        assert exported[0]["metadata"] == {"x": 1}
        assert exported[0]["tags"] == ["tag"]
    
    def test_file_persistence(self):
        """Test that data persists to file."""
        from memory.storage import VectorStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create and save
            storage1 = VectorStorage(db_path=db_path)
            embedding = np.random.randn(384).astype(np.float32)
            storage1.save(key="persist", content="Persisted", embedding=embedding)
            storage1.close()
            
            # Reopen and verify
            storage2 = VectorStorage(db_path=db_path)
            item = storage2.get("persist")
            assert item is not None
            assert item.content == "Persisted"


# ============================================================================
# Memory Manager Tests
# ============================================================================

class TestMemoryManager:
    """Tests for the memory manager."""
    
    def test_manager_init(self):
        """Test manager initialization."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        assert manager is not None
    
    def test_save_and_search(self):
        """Test saving and searching."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        manager.save(content="The quick brown fox jumps over the lazy dog")
        manager.save(content="Python is a programming language")
        manager.save(content="Machine learning uses neural networks")
        
        results = manager.search("programming", top_k=3)
        assert len(results) > 0
        
        # Programming query should find Python content
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)
    
    def test_save_with_key(self):
        """Test saving with specific key."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        key = manager.save(content="Content", key="mykey")
        assert key == "mykey"
        
        item = manager.get("mykey")
        assert item is not None
    
    def test_save_batch(self):
        """Test batch saving."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        items = [
            {"content": "First item", "tags": ["a"]},
            {"content": "Second item", "tags": ["b"]},
            {"content": "Third item", "metadata": {"x": 1}},
        ]
        
        keys = manager.save_batch(items, show_progress=False)
        assert len(keys) == 3
        
        for key in keys:
            assert manager.get(key) is not None
    
    def test_search_with_filters(self):
        """Test search with metadata and tag filters."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        manager.save(content="Work meeting notes", tags=["work"])
        manager.save(content="Personal diary entry", tags=["personal"])
        manager.save(content="Project documentation", metadata={"type": "docs"})
        
        # Filter by tags
        results = manager.search("notes", tags_filter=["work"])
        contents = [r.content for r in results]
        assert all("personal" not in c.lower() for c in contents)
    
    def test_update_operations(self):
        """Test update operations."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        key = manager.save(content="Test", metadata={"a": 1})
        
        manager.update_metadata(key, {"b": 2})
        item = manager.get(key)
        assert item.metadata == {"a": 1, "b": 2}
        
        manager.update_tags(key, ["new_tag"])
        item = manager.get(key)
        assert "new_tag" in item.tags
    
    def test_similarity(self):
        """Test similarity calculation."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        sim = manager.similarity(
            "The cat sat on the mat",
            "A feline rested on a rug"
        )
        assert 0 <= sim <= 1
    
    def test_get_stats(self):
        """Test getting statistics."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        manager.save(content="Test content")
        
        stats = manager.get_stats()
        assert stats["total_memories"] >= 1
        assert "embedding_model" in stats
    
    def test_export_import(self):
        """Test export and import."""
        from memory.memory import MemoryManager
        
        manager1 = MemoryManager(db_path=":memory:", use_gpu=False)
        manager1.save(content="Export test", key="export1", tags=["test"])
        
        exported = manager1.export_all()
        
        manager2 = MemoryManager(db_path=":memory:", use_gpu=False)
        count = manager2.import_batch(exported)
        
        assert count == 1
        item = manager2.get("export1")
        assert item is not None
        assert item.content == "Export test"
    
    def test_text_search(self):
        """Test full-text search."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        manager.save(content="The quick brown fox")
        manager.save(content="Lazy dog sleeps")
        
        results = manager.search_text("fox")
        assert len(results) > 0
        assert any("fox" in item.content for item in results)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full system."""
    
    def test_full_workflow(self):
        """Test complete workflow."""
        from memory.memory import MemoryManager
        
        manager = MemoryManager(db_path=":memory:", use_gpu=False)
        
        # Save some memories
        key1 = manager.save(
            content="Meeting with John about Q4 planning",
            tags=["meetings", "work"],
            metadata={"attendees": ["John"]}
        )
        
        key2 = manager.save(
            content="Buy groceries: milk, eggs, bread",
            tags=["todos", "personal"]
        )
        
        key3 = manager.save(
            content="Quarterly review preparation notes",
            tags=["work"],
            metadata={"type": "notes"}
        )
        
        # Search
        results = manager.search("Q4 planning meeting")
        assert len(results) > 0
        
        # Search with filter
        work_results = manager.search("notes", tags_filter=["work"])
        for r in work_results:
            assert "work" in r.tags
        
        # Update
        manager.update_tags(key1, ["important"], merge=True)
        item = manager.get(key1)
        assert "important" in item.tags
        
        # Delete
        manager.delete(key2)
        assert manager.get(key2) is None
        
        # Stats
        stats = manager.get_stats()
        assert stats["total_memories"] == 2
    
    def test_concurrent_access(self):
        """Test concurrent access with threading."""
        import threading
        from memory.memory import MemoryManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "concurrent.db")
            manager = MemoryManager(db_path=db_path, use_gpu=False)
            
            errors = []
            
            def save_memories(thread_id):
                try:
                    for i in range(5):
                        manager.save(
                            content=f"Thread {thread_id} memory {i}",
                            key=f"t{thread_id}_m{i}"
                        )
                except Exception as e:
                    errors.append(e)
            
            threads = [threading.Thread(target=save_memories, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(errors) == 0
            assert manager.count() == 15


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def memory_manager():
    """Create a memory manager for testing."""
    from memory.memory import MemoryManager
    manager = MemoryManager(db_path=":memory:", use_gpu=False)
    yield manager
    manager.close()


@pytest.fixture
def storage():
    """Create a storage instance for testing."""
    from memory.storage import VectorStorage
    storage = VectorStorage(db_path=":memory:")
    yield storage
    storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
