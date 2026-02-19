"""
Advanced Vector Storage with sqlite-vec and HNSW for high-performance similarity search.

Features:
- Zero server dependencies (embedded SQLite)
- AVX/NEON SIMD acceleration via sqlite-vec
- HNSW ANN index for O(log n) vector search (100x faster)
- Native SQL vector operations
- Thread-safe connection pooling
- Metadata filtering with JSON operators
- TTL (time-to-live) support
- Full-text search fallback
- Hybrid search: HNSW + sqlite-vec + FTS5
"""
from __future__ import annotations

import logging
import sqlite3
import struct
import json
import threading
from pathlib import Path
from queue import Queue
from typing import Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import numpy as np

# Lazy import HNSW
try:
    from .hnsw_index import HNSWIndex, HNSW_AVAILABLE
except ImportError:
    HNSW_AVAILABLE = False
    HNSWIndex = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A memory item with content and metadata."""
    id: int
    key: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    expires_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except (ValueError, TypeError):
            return False


def serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize numpy array to sqlite-vec float32 format."""
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return vector.tobytes()


def deserialize_f32(blob: bytes) -> np.ndarray:
    """Deserialize sqlite-vec float32 blob to numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._semaphore = threading.Semaphore(pool_size)
        self._lock = threading.Lock()
        self._vec_loaded = False
        
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        
        # Load sqlite-vec
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_loaded = True
        except Exception:
            pass  # sqlite_vec is optional
            
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (bounded by semaphore)."""
        self._semaphore.acquire(timeout=10)
        conn = None
        try:
            try:
                conn = self._pool.get_nowait()
            except Exception:
                conn = self._create_connection()
            yield conn
        finally:
            if conn:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    conn.close()
            self._semaphore.release()

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                break


class VectorStorage:
    """
    Advanced SQLite-vec based vector storage with HNSW ANN indexing.
    
    Features:
    - Thread-safe connection pooling
    - WAL mode for concurrent reads
    - HNSW index for O(log n) vector search
    - Metadata filtering
    - TTL support
    - Tag-based organization
    """
    
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        dimension: int = 384,
        pool_size: int = 5,
        use_hnsw: bool = True,
        hnsw_max_elements: int = 100_000,
        hnsw_ef_construction: int = 200,
        hnsw_M: int = 16,
    ):
        """
        Initialize vector storage.
        
        Args:
            db_path: Path to SQLite database (or ":memory:" for in-memory)
            dimension: Vector embedding dimension
            pool_size: Connection pool size for concurrent access
            use_hnsw: Enable HNSW ANN index for fast search
            hnsw_max_elements: Maximum HNSW index capacity
            hnsw_ef_construction: HNSW construction parameter (higher = better quality)
            hnsw_M: HNSW links per node (higher = better quality)
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self.dimension = dimension
        self._pool: Optional[ConnectionPool] = None
        self._single_conn: Optional[sqlite3.Connection] = None
        self._is_memory = db_path == ":memory:"
        self._lock = threading.RLock()
        self._vec_available = False
        
        # HNSW index
        self._use_hnsw = use_hnsw and HNSW_AVAILABLE
        self._hnsw: Optional['HNSWIndex'] = None
        self._hnsw_config = {
            'max_elements': hnsw_max_elements,
            'ef_construction': hnsw_ef_construction,
            'M': hnsw_M,
        }
        
        # Create parent directory if needed
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # Initialize HNSW index
            if self._use_hnsw:
                self._init_hnsw()
    
    def _init_hnsw(self):
        """Initialize HNSW index."""
        if not self._use_hnsw or not HNSW_AVAILABLE:
            return
        
        try:
            # HNSW index path alongside SQLite database
            hnsw_path = None
            if isinstance(self.db_path, Path):
                hnsw_path = self.db_path.with_suffix('.hnsw')
            
            self._hnsw = HNSWIndex(
                dimension=self.dimension,
                max_elements=self._hnsw_config['max_elements'],
                ef_construction=self._hnsw_config['ef_construction'],
                M=self._hnsw_config['M'],
                space='cosine',
                index_path=hnsw_path,
            )
            logger.info("✓ HNSW index initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HNSW index: {e}")
            self._use_hnsw = False
            self._hnsw = None
    
    def rebuild_hnsw_index(self):
        """Rebuild HNSW index from all embeddings in database."""
        if not self._use_hnsw or not self._hnsw:
            return
        
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT id, embedding FROM memories 
                WHERE embedding IS NOT NULL
            """).fetchall()
            
            if not rows:
                return
            
            ids = [row['id'] for row in rows]
            embeddings = np.array([
                deserialize_f32(row['embedding']) for row in rows
            ], dtype=np.float32)
            
            self._hnsw.add_batch(ids, embeddings)
            logger.info(f"✓ HNSW index rebuilt with {len(ids)} vectors")
    
    @contextmanager
    def _get_conn(self):
        """Get a database connection (thread-safe)."""
        if self._is_memory:
            # In-memory DB uses single connection
            if self._single_conn is None:
                with self._lock:
                    if self._single_conn is None:
                        self._single_conn = self._create_single_connection()
                        self._init_schema(self._single_conn)
            yield self._single_conn
        else:
            # File DB uses connection pool
            if self._pool is None:
                with self._lock:
                    if self._pool is None:
                        db_str = str(self.db_path)
                        self._pool = ConnectionPool(db_str)
                        # Initialize schema with first connection
                        with self._pool.get_connection() as conn:
                            self._init_schema(conn)
                        self._vec_available = self._pool._vec_loaded
                        logger.info(f"✓ Connected to {db_str}")
            
            with self._pool.get_connection() as conn:
                yield conn
    
    def _create_single_connection(self) -> sqlite3.Connection:
        """Create a single connection for in-memory DB."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_available = True
            logger.info("✓ sqlite-vec extension loaded")
        except Exception as e:
            logger.warning(f"sqlite-vec not available: {e}")
        
        return conn
    
    def _init_schema(self, conn: sqlite3.Connection):
        """Initialize database schema."""
        conn.executescript("""
            -- Main memory table with extended fields
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                expires_at TEXT DEFAULT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed_at TEXT DEFAULT NULL
            );
            
            -- Tags junction table for O(1) tag lookups
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON memory_tags(tag);
            
            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
            CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
            CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);
            
            -- Full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, content, content='memories', content_rowid='id'
            );
            
            -- FTS sync triggers
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, key, content) VALUES (new.id, new.key, new.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content) VALUES ('delete', old.id, old.key, old.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content) VALUES ('delete', old.id, old.key, old.content);
                INSERT INTO memories_fts(rowid, key, content) VALUES (new.id, new.key, new.content);
            END;
        """)
        
        # Migrate: populate memory_tags from JSON tags column for existing data
        try:
            conn.execute("""
                INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                SELECT m.id, j.value
                FROM memories m, json_each(m.tags) j
                WHERE m.tags IS NOT NULL AND m.tags != '[]'
            """)
        except Exception as e:
            logger.debug(f"Tags migration skipped (may be first run): {e}")
        
        # Create vector index if sqlite-vec is available
        try:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    embedding float[{self.dimension}]
                );
            """)
            logger.info(f"✓ Vector index created (dim={self.dimension})")
        except Exception as e:
            logger.debug(f"Vector index not available: {e}")
        
        conn.commit()
    
    def save(
        self,
        key: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> int:
        """
        Save a memory item.
        
        Args:
            key: Unique key for the memory
            content: Text content
            embedding: Vector embedding (optional)
            metadata: Additional metadata (optional)
            tags: List of tags for organization (optional)
            ttl_seconds: Time-to-live in seconds (optional)
            
        Returns:
            ID of the saved memory
        """
        embedding_blob = serialize_f32(embedding) if embedding is not None else None
        metadata_str = json.dumps(metadata or {})
        tags_str = json.dumps(tags or [])
        
        # Calculate expiration time
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO memories (key, content, embedding, metadata, tags, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    content = excluded.content,
                    embedding = excluded.embedding,
                    metadata = excluded.metadata,
                    tags = excluded.tags,
                    expires_at = excluded.expires_at,
                    updated_at = datetime('now')
                RETURNING id
            """, (key, content, embedding_blob, metadata_str, tags_str, expires_at))
            
            row_id = cursor.fetchone()[0]
            
            # Sync tags junction table
            tag_list = tags or []
            conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (row_id,))
            if tag_list:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(row_id, t) for t in tag_list]
                )
            
            # Update vector index
            if embedding is not None:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO memories_vec (rowid, embedding)
                        VALUES (?, ?)
                    """, (row_id, embedding_blob))
                except Exception as e:
                    logger.warning(f"Failed to update sqlite-vec index for key={key}: {e}")

                # Update HNSW index (C2: wrapped with lock)
                if self._use_hnsw and self._hnsw:
                    with self._lock:
                        try:
                            self._hnsw.add(row_id, embedding)
                        except Exception as e:
                            logger.warning(f"Failed to add to HNSW index for key={key}: {e}")
            
            conn.commit()
            return row_id
    
    def save_batch(
        self,
        items: list[dict],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[int]:
        """
        Save multiple memory items in a single transaction.
        
        Args:
            items: List of dicts with keys: key, content, metadata, tags, ttl_seconds
            embeddings: Optional array of embeddings (n_items, dimension)
            
        Returns:
            List of saved memory IDs
        """
        ids = []
        
        with self._get_conn() as conn:
            for i, item in enumerate(items):
                embedding = embeddings[i] if embeddings is not None else None
                embedding_blob = serialize_f32(embedding) if embedding is not None else None
                metadata_str = json.dumps(item.get("metadata", {}))
                tags_str = json.dumps(item.get("tags", []))
                
                expires_at = None
                if item.get("ttl_seconds"):
                    expires_at = (datetime.now() + timedelta(seconds=item["ttl_seconds"])).isoformat()
                
                cursor = conn.execute("""
                    INSERT INTO memories (key, content, embedding, metadata, tags, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        content = excluded.content,
                        embedding = excluded.embedding,
                        metadata = excluded.metadata,
                        tags = excluded.tags,
                        expires_at = excluded.expires_at,
                        updated_at = datetime('now')
                    RETURNING id
                """, (item["key"], item["content"], embedding_blob, metadata_str, tags_str, expires_at))
                
                row_id = cursor.fetchone()[0]
                ids.append(row_id)
                
                # Sync tags junction table
                tag_list = item.get("tags", [])
                conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (row_id,))
                if tag_list:
                    conn.executemany(
                        "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        [(row_id, t) for t in tag_list]
                    )
                
                # Update vector index
                if embedding is not None:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO memories_vec (rowid, embedding)
                            VALUES (?, ?)
                        """, (row_id, embedding_blob))
                    except Exception as e:
                        logger.warning(f"Failed to update sqlite-vec index for row_id={row_id}: {e}")

            conn.commit()

        # Update HNSW index in batch (C2: wrapped with lock)
        if self._use_hnsw and self._hnsw and embeddings is not None:
            with self._lock:
                try:
                    self._hnsw.add_batch(ids, embeddings)
                except Exception as e:
                    logger.warning(f"Failed to add batch to HNSW index: {e}")
        
        return ids
    
    def _parse_row(self, row, include_embedding: bool = False) -> MemoryItem:
        """Parse a database row into MemoryItem.
        
        Args:
            row: sqlite3.Row object
            include_embedding: If True, deserialize embedding blob. Default False
                             to avoid expensive deserialization in list/search paths.
        """
        tags = []
        try:
            tags = json.loads(row['tags']) if row['tags'] else []
        except (json.JSONDecodeError, TypeError):
            pass
        
        embedding = None
        if include_embedding:
            try:
                raw = row['embedding']
                embedding = deserialize_f32(raw) if raw else None
            except (IndexError, KeyError):
                pass
            
        return MemoryItem(
            id=row['id'],
            key=row['key'],
            content=row['content'],
            embedding=embedding,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            expires_at=row['expires_at'] if 'expires_at' in row.keys() else None,
            tags=tags,
        )
    
    def get(self, key: str, update_access: bool = True) -> Optional[MemoryItem]:
        """
        Get a memory by key.
        
        Args:
            key: Memory key
            update_access: Whether to update access count and timestamp
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
            
            if row is None:
                return None
            
            item = self._parse_row(row)

            # Check expiration - inline delete to avoid nested connection acquisition
            if item.is_expired:
                row_id = row['id']
                conn.execute("DELETE FROM memories WHERE id = ?", (row_id,))
                try:
                    conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row_id,))
                except Exception as e:
                    logger.warning(f"Failed to delete expired vector entry {row_id}: {e}")
                conn.commit()
                return None
            
            # Update access stats
            if update_access:
                conn.execute("""
                    UPDATE memories SET 
                        access_count = access_count + 1,
                        last_accessed_at = datetime('now')
                    WHERE key = ?
                """, (key,))
                conn.commit()
            
            return item
    
    def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE key = ?", (key,)
            ).fetchone()
            
            if row is None:
                return False
            
            row_id = row['id']
            
            # Delete from main table (triggers handle FTS, CASCADE handles memory_tags)
            conn.execute("DELETE FROM memories WHERE id = ?", (row_id,))
            # Explicit delete from junction table (CASCADE may not work in all SQLite builds)
            conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (row_id,))
            
            # Delete from vector index
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row_id,))
            except Exception as e:
                logger.warning(f"Failed to delete from sqlite-vec index for row_id={row_id}: {e}")
            
            # Sync HNSW index
            if self._use_hnsw and self._hnsw:
                try:
                    self._hnsw.delete(row_id)
                except Exception as e:
                    logger.debug(f"Failed to delete from HNSW for row_id={row_id}: {e}")
            
            conn.commit()
            return True
    
    def delete_expired(self) -> int:
        """Delete all expired memories. Returns count deleted."""
        with self._get_conn() as conn:
            # Direct comparison — ISO 8601 strings are lexicographically ordered
            rows = conn.execute("""
                SELECT id FROM memories
                WHERE expires_at IS NOT NULL
                AND expires_at < datetime('now')
            """).fetchall()

            if not rows:
                return 0

            ids = [r['id'] for r in rows]

            # Delete from main table
            placeholders = ','.join('?' * len(ids))
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            
            # Delete from tags junction table
            conn.execute(f"DELETE FROM memory_tags WHERE memory_id IN ({placeholders})", ids)

            # Delete from vector index
            try:
                conn.execute(f"DELETE FROM memories_vec WHERE rowid IN ({placeholders})", ids)
            except Exception as e:
                logger.warning(f"Failed to delete {len(ids)} expired entries from sqlite-vec: {e}")
            
            # Sync HNSW index
            if self._use_hnsw and self._hnsw:
                for row_id in ids:
                    try:
                        self._hnsw.delete(row_id)
                    except Exception:
                        pass

            conn.commit()
            return len(ids)
    
    def search_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
        exclude_expired: bool = True,
        use_hnsw: Optional[bool] = None,
        hnsw_ef: int = 50,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Search memories by vector similarity with optional filters.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            metadata_filter: Filter by metadata fields (exact match)
            tags_filter: Filter by tags (any match)
            exclude_expired: Whether to exclude expired memories
            use_hnsw: Force HNSW usage (None=auto, True=force HNSW, False=force sqlite-vec)
            hnsw_ef: HNSW search parameter (higher = better quality, slower)
            
        Returns:
            List of (MemoryItem, similarity_score) tuples
        """
        # Determine search method
        should_use_hnsw = use_hnsw if use_hnsw is not None else (self._use_hnsw and self._hnsw is not None)
        
        # Try HNSW search first (O(log n) - 100x faster)
        if should_use_hnsw and self._hnsw:
            try:
                return self._search_hnsw(
                    query_embedding, top_k, threshold,
                    metadata_filter, tags_filter, exclude_expired, hnsw_ef
                )
            except Exception as e:
                logger.debug(f"HNSW search failed, falling back: {e}")
        
        # Fallback to sqlite-vec or pure Python
        query_blob = serialize_f32(query_embedding)
        
        with self._get_conn() as conn:
            # Try sqlite-vec KNN search
            try:
                # Get more candidates to account for filtering
                fetch_k = top_k * 3 if (metadata_filter or tags_filter) else top_k
                
                rows = conn.execute("""
                    SELECT 
                        m.*,
                        v.distance
                    FROM memories_vec v
                    JOIN memories m ON m.id = v.rowid
                    WHERE v.embedding MATCH ?
                    ORDER BY v.distance ASC
                    LIMIT ?
                """, (query_blob, fetch_k)).fetchall()
                
                results = []
                for row in rows:
                    # Convert distance to similarity
                    similarity = 1.0 - row['distance']
                    if similarity < threshold:
                        continue
                    
                    item = self._parse_row(row)
                    
                    # Apply filters
                    if exclude_expired and item.is_expired:
                        continue
                    
                    if metadata_filter:
                        match = all(
                            item.metadata.get(k) == v 
                            for k, v in metadata_filter.items()
                        )
                        if not match:
                            continue
                    
                    if tags_filter:
                        if not any(t in item.tags for t in tags_filter):
                            continue
                    
                    results.append((item, similarity))
                    
                    if len(results) >= top_k:
                        break
                
                return results
                
            except Exception as e:
                logger.debug(f"Vector search failed, using fallback: {e}")
                return self._search_vector_fallback(
                    query_embedding, top_k, threshold,
                    metadata_filter, tags_filter, exclude_expired
                )
    
    def _search_hnsw(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
        exclude_expired: bool = True,
        ef: int = 50,
    ) -> list[tuple[MemoryItem, float]]:
        """Fast HNSW-based vector search with post-filtering."""
        # Get more candidates for filtering
        fetch_k = top_k * 3 if (metadata_filter or tags_filter or exclude_expired) else top_k
        
        # HNSW search (O(log n)) - C2: wrapped with lock for thread safety
        with self._lock:
            hnsw_results = self._hnsw.search(query_embedding, k=fetch_k, ef=ef)
        
        if not hnsw_results:
            return []
        
        # Get candidate IDs
        candidate_ids = [r[0] for r in hnsw_results]
        similarity_map = {r[0]: r[1] for r in hnsw_results}
        
        # Fetch full records from database
        with self._get_conn() as conn:
            placeholders = ','.join('?' * len(candidate_ids))
            rows = conn.execute(f"""
                SELECT id, key, content, metadata, tags, created_at, updated_at, expires_at, access_count, last_accessed_at
                FROM memories WHERE id IN ({placeholders})
            """, candidate_ids).fetchall()
            
            # Create ID to row mapping
            rows_by_id = {row['id']: row for row in rows}
            
            # Build results in similarity order
            results = []
            for row_id, similarity in hnsw_results:
                if similarity < threshold:
                    continue
                
                if row_id not in rows_by_id:
                    continue
                
                item = self._parse_row(rows_by_id[row_id])
                
                # Apply filters
                if exclude_expired and item.is_expired:
                    continue
                
                if metadata_filter:
                    match = all(
                        item.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                if tags_filter:
                    if not any(t in item.tags for t in tags_filter):
                        continue
                
                results.append((item, similarity))
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def _search_vector_fallback(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
        exclude_expired: bool = True,
    ) -> list[tuple[MemoryItem, float]]:
        """Fallback vector search using pure Python cosine similarity."""
        # H2: Apply a cap to avoid loading unbounded vectors into memory
        scan_limit = max(top_k * 100, 10000)
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE embedding IS NOT NULL LIMIT ?",
                (scan_limit,)
            ).fetchall()
            
            if not rows:
                return []
            
            # Calculate similarities
            results = []
            for row in rows:
                embedding = deserialize_f32(row['embedding'])
                similarity = float(np.dot(query_embedding, embedding))
                
                if similarity < threshold:
                    continue
                
                item = self._parse_row(row)
                
                # Apply filters
                if exclude_expired and item.is_expired:
                    continue
                
                if metadata_filter:
                    match = all(
                        item.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                if tags_filter:
                    if not any(t in item.tags for t in tags_filter):
                        continue
                
                results.append((item, similarity))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Sanitize a query string for FTS5 MATCH.
        
        FTS5 treats `-` as NOT operator and other punctuation as syntax.
        This wraps each token in double quotes to treat them as literals.
        e.g. "anti-cheat" -> '"anti-cheat"'  (exact phrase)
             "UE4SS hook" -> '"UE4SS" "hook"'  (AND of literals)
        """
        import re
        query = query.strip()
        if not query:
            return query
        # If user already quoted, pass through
        if query.startswith('"') and query.endswith('"'):
            return query
        # Split on whitespace, quote each token to escape operators
        tokens = query.split()
        return " ".join(f'"{t}"' for t in tokens if t)

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
    ) -> list[MemoryItem]:
        """Search memories by full-text search with optional filters."""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT m.* FROM memories m
                JOIN memories_fts f ON m.id = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (safe_query, top_k * 2)).fetchall()
            
            results = []
            for row in rows:
                item = self._parse_row(row)
                
                if item.is_expired:
                    continue
                
                if metadata_filter:
                    match = all(
                        item.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                if tags_filter:
                    if not any(t in item.tags for t in tags_filter):
                        continue
                
                results.append(item)
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def _search_text_ranked(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Internal FTS search returning (item, rank) for RRF fusion."""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT m.*, bm25(memories_fts) AS bm25_score
                FROM memories m
                JOIN memories_fts f ON m.id = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
            """, (safe_query, top_k * 2)).fetchall()
            
            results = []
            for i, row in enumerate(rows):
                item = self._parse_row(row)
                
                if item.is_expired:
                    continue
                
                if metadata_filter:
                    match = all(
                        item.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                if tags_filter:
                    if not any(t in item.tags for t in tags_filter):
                        continue
                
                # BM25 score (lower is better, so we use rank position)
                # Normalize: rank 0 -> 1.0, rank increases -> score decreases
                rank_score = 1.0 / (i + 1)
                results.append((item, rank_score))
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        metadata_filter: Optional[dict] = None,
        tags_filter: Optional[list[str]] = None,
        exclude_expired: bool = True,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Hybrid search combining vector similarity and BM25 text search using RRF.
        
        Reciprocal Rank Fusion (RRF) combines multiple ranking lists by:
        RRF(d) = Σ 1 / (k + rank_i(d))
        
        This provides +15-30% accuracy improvement over vector-only search.
        
        Args:
            query: Text query for BM25 search
            query_embedding: Query vector for similarity search
            top_k: Number of results to return
            threshold: Minimum similarity score for vector search
            metadata_filter: Filter by metadata fields
            tags_filter: Filter by tags
            exclude_expired: Whether to exclude expired memories
            vector_weight: Weight for vector search results (0-1)
            text_weight: Weight for text search results (0-1)
            rrf_k: RRF constant (default 60, higher = more conservative)
            
        Returns:
            List of (MemoryItem, combined_score) tuples sorted by score
        """
        # Get vector search results
        vector_results = self.search_vector(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for fusion
            threshold=threshold,
            metadata_filter=metadata_filter,
            tags_filter=tags_filter,
            exclude_expired=exclude_expired,
        )
        
        # Get text search results
        try:
            text_results = self._search_text_ranked(
                query=query,
                top_k=top_k * 2,
                metadata_filter=metadata_filter,
                tags_filter=tags_filter,
            )
        except Exception as e:
            logger.debug(f"Text search failed, using vector only: {e}")
            text_results = []
        
        # If only one source has results, return it
        if not vector_results and not text_results:
            return []
        if not text_results:
            return vector_results[:top_k]
        if not vector_results:
            return text_results[:top_k]
        
        # Apply RRF fusion
        return self._rrf_fusion(
            vector_results=vector_results,
            text_results=text_results,
            top_k=top_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
            rrf_k=rrf_k,
        )
    
    def _rrf_fusion(
        self,
        vector_results: list[tuple[MemoryItem, float]],
        text_results: list[tuple[MemoryItem, float]],
        top_k: int,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Reciprocal Rank Fusion to combine multiple ranked lists.
        
        RRF(d) = Σ weight_i / (k + rank_i(d))
        
        Args:
            vector_results: Results from vector search [(item, score), ...]
            text_results: Results from text search [(item, score), ...]
            top_k: Number of results to return
            vector_weight: Weight for vector results
            text_weight: Weight for text results
            rrf_k: RRF constant (typically 60)
            
        Returns:
            Fused results sorted by combined RRF score
        """
        # Build ID -> item mapping and RRF scores
        items_by_id: dict[int, MemoryItem] = {}
        rrf_scores: dict[int, float] = {}
        
        # Process vector results
        for rank, (item, score) in enumerate(vector_results):
            items_by_id[item.id] = item
            rrf_score = vector_weight / (rrf_k + rank + 1)
            rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + rrf_score
        
        # Process text results
        for rank, (item, score) in enumerate(text_results):
            items_by_id[item.id] = item
            rrf_score = text_weight / (rrf_k + rank + 1)
            rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + rrf_score
        
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        results = []
        for item_id in sorted_ids[:top_k]:
            item = items_by_id[item_id]
            score = rrf_scores[item_id]
            results.append((item, score))
        
        return results

    
    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        tags_filter: Optional[list[str]] = None,
        exclude_expired: bool = True,
    ) -> list[MemoryItem]:
        """List all memories with optional filters."""
        with self._get_conn() as conn:
            if exclude_expired:
                rows = conn.execute("""
                    SELECT id, key, content, metadata, tags, created_at, updated_at, expires_at, access_count, last_accessed_at
                    FROM memories
                    WHERE expires_at IS NULL OR expires_at > datetime('now')
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, key, content, metadata, tags, created_at, updated_at, expires_at, access_count, last_accessed_at
                    FROM memories
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset)).fetchall()
            
            results = []
            for row in rows:
                item = self._parse_row(row)
                
                if tags_filter:
                    if not any(t in item.tags for t in tags_filter):
                        continue
                
                results.append(item)
            
            return results
    
    def list_by_tags(self, tags: list[str], limit: int = 100) -> list[MemoryItem]:
        """List memories that have any of the specified tags (uses junction table)."""
        if not tags:
            return []
        with self._get_conn() as conn:
            placeholders = ','.join('?' * len(tags))
            rows = conn.execute(f"""
                SELECT DISTINCT m.id, m.key, m.content, m.metadata, m.tags,
                       m.created_at, m.updated_at, m.expires_at, m.access_count, m.last_accessed_at
                FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                WHERE mt.tag IN ({placeholders})
                  AND (m.expires_at IS NULL OR m.expires_at > datetime('now'))
                ORDER BY m.updated_at DESC
                LIMIT ?
            """, (*tags, limit)).fetchall()
            
            return [self._parse_row(row) for row in rows]
    
    def count(self, exclude_expired: bool = True) -> int:
        """Get total number of memories."""
        with self._get_conn() as conn:
            if exclude_expired:
                return conn.execute("""
                    SELECT COUNT(*) FROM memories
                    WHERE expires_at IS NULL OR expires_at > datetime('now')
                """).fetchone()[0]
            else:
                return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            with_embedding = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            expired = conn.execute("""
                SELECT COUNT(*) FROM memories
                WHERE expires_at IS NOT NULL AND expires_at < datetime('now')
            """).fetchone()[0]
            
            return {
                "total_memories": total,
                "with_embedding": with_embedding,
                "expired": expired,
                "active": total - expired,
                "dimension": self.dimension,
                "vec_available": self._vec_available,
            }
    
    def update_metadata(self, key: str, metadata: dict, merge: bool = True) -> bool:
        """
        Update memory metadata.
        
        Args:
            key: Memory key
            metadata: New metadata
            merge: If True, merge with existing. If False, replace.
        """
        with self._get_conn() as conn:
            if merge:
                row = conn.execute(
                    "SELECT metadata FROM memories WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    return False
                existing = json.loads(row['metadata']) if row['metadata'] else {}
                existing.update(metadata)
                metadata = existing
            
            conn.execute("""
                UPDATE memories SET 
                    metadata = ?,
                    updated_at = datetime('now')
                WHERE key = ?
            """, (json.dumps(metadata), key))
            conn.commit()
            return conn.total_changes > 0
    
    def update_tags(self, key: str, tags: list[str], merge: bool = True) -> bool:
        """
        Update memory tags.
        
        Args:
            key: Memory key
            tags: New tags
            merge: If True, add to existing. If False, replace.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, tags FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            
            memory_id = row['id']
            
            if merge:
                existing = json.loads(row['tags']) if row['tags'] else []
                tags = list(set(existing + tags))
            
            conn.execute("""
                UPDATE memories SET 
                    tags = ?,
                    updated_at = datetime('now')
                WHERE key = ?
            """, (json.dumps(tags), key))
            
            # Sync junction table
            conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            if tags:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(memory_id, t) for t in tags]
                )
            
            conn.commit()
            return True
    
    def export_all(self) -> list[dict]:
        """Export all memories as a list of dictionaries."""
        with self._get_conn() as conn:
            # H1: Use fetchmany to avoid loading all rows into memory at once
            cursor = conn.execute("SELECT * FROM memories")
            result = []
            while True:
                rows = cursor.fetchmany(10000)
                if not rows:
                    break
                for row in rows:
                    result.append({
                        "key": row['key'],
                        "content": row['content'],
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                        "tags": json.loads(row['tags']) if row['tags'] else [],
                        "created_at": row['created_at'],
                        "updated_at": row['updated_at'],
                        "expires_at": row['expires_at'],
                    })
            if len(result) > 50000:
                logger.warning(f"export_all() returned {len(result)} records; consider using pagination for large datasets")
            return result
    
    def close(self):
        """Close database connections and save HNSW index."""
        # Save HNSW index before closing
        if self._hnsw:
            try:
                self._hnsw.save()
                self._hnsw.close()
            except Exception as e:
                logger.debug(f"Failed to save HNSW index: {e}")
            self._hnsw = None
        
        if self._single_conn:
            self._single_conn.close()
            self._single_conn = None
        if self._pool:
            self._pool.close_all()
            self._pool = None
