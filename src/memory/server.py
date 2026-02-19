"""
Advanced MCP Server for Ultra-Light Memory.

Provides comprehensive memory management tools:
- Semantic search with filtering
- Batch operations
- TTL support
- Tag-based organization
- Import/export
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from .memory import get_manager, MemoryManager

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("ultra-light-memory")

# Manager instance (initialized on first use)
_manager: MemoryManager | None = None
_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get the memory manager instance."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                db_path = os.environ.get(
                    "MEMORY_DB_PATH",
                    os.path.expanduser("~/.ultra-light-memory/memory.db")
                )
                use_gpu = os.environ.get("MEMORY_USE_GPU", "true").lower() in {"true", "1", "yes", "on"}
                model_key = os.environ.get("MEMORY_MODEL", None)

                _manager = get_manager(db_path=db_path, use_gpu=use_gpu, model_key=model_key)
                logger.info(f"Memory manager initialized: {db_path}")

    return _manager


# ============================================================================
# Consolidated MCP Tools (9 total)
# ============================================================================

@mcp.tool()
def memory_save(
    content: str = "",
    key: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    ttl_seconds: int | None = None,
    items: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Save one or multiple memories.

    Single-save mode (items is None or empty):
        content     : Text content to save (required)
        key         : Optional unique key (auto-generated if omitted)
        metadata    : Optional metadata dictionary
        tags        : Optional list of tags for organization
        ttl_seconds : Optional TTL in seconds; memory expires after this

    Batch-save mode (items is a non-empty list):
        items : List of memory dicts, each with:
                  content (required), key (optional), metadata (optional),
                  tags (optional), ttl_seconds (optional)
        All other parameters are ignored in batch mode.

    Returns:
        Single-save  → {"success": True, "key": <key>, "message": ...}
        Batch-save   → {"success": True, "count": N, "keys": [...], "message": ...}
    """
    manager = get_memory_manager()

    try:
        if items:
            keys = manager.save_batch(items)
            return {
                "success": True,
                "count": len(keys),
                "keys": keys,
                "message": f"Saved {len(keys)} memories",
            }
        saved_key = manager.save(
            content=content,
            key=key,
            metadata=metadata,
            tags=tags,
            ttl_seconds=ttl_seconds,
        )
        return {
            "success": True,
            "key": saved_key,
            "message": f"Memory saved with key: {saved_key}",
        }
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_get(key: str) -> dict:
    """
    Get a specific memory by key.
    
    Args:
        key: Unique key of the memory
        
    Returns:
        Dictionary with memory content or error
    """
    manager = get_memory_manager()
    
    try:
        item = manager.get(key)
        if item is None:
            return {"success": False, "error": f"Memory not found: {key}"}
        
        return {
            "success": True,
            "key": item.key,
            "content": item.content,
            "metadata": item.metadata,
            "tags": item.tags,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "expires_at": item.expires_at,
        }
    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_delete(
    key: str = "",
    by: str = "key",
    tag: str = "",
    prefix: str = "",
) -> dict:
    """
    Delete one or more memories using different strategies.

    by="key" (default):
        key : Unique key of the memory to delete (required)

    by="expired":
        Deletes all memories whose TTL has elapsed.
        No other parameters needed.

    by="tag":
        tag : Tag string — all memories containing this tag are deleted.

    by="prefix":
        prefix : Key prefix — all memories whose key starts with this are deleted.
                 Useful for removing all chunks of a file import.

    Returns:
        by=key     → {"success": bool, "message": ...}
        by=expired → {"success": True, "deleted_count": N, "message": ...}
        by=tag/prefix → {"success": True, "deleted": N, "sample_keys": [...], "message": ...}
    """
    manager = get_memory_manager()

    try:
        if by == "key":
            deleted = manager.delete(key)
            return {
                "success": deleted,
                "message": f"Memory {'deleted' if deleted else 'not found'}: {key}",
            }

        if by == "expired":
            count = manager.delete_expired()
            return {
                "success": True,
                "deleted_count": count,
                "message": f"Deleted {count} expired memories",
            }

        storage = manager.storage

        if by == "tag":
            if not tag:
                return {"success": False, "error": "tag parameter required when by='tag'"}
            with storage._get_conn() as conn:
                escaped = tag.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                rows = conn.execute(
                    "SELECT id, key FROM memories WHERE tags LIKE ? ESCAPE '\\'",
                    (f'%"{escaped}"%',),
                ).fetchall()
                if not rows:
                    return {"success": True, "deleted": 0, "message": f"No memories found with tag: {tag}"}
                ids = [r['id'] for r in rows]
                keys = [r['key'] for r in rows]
                ph = ','.join('?' * len(ids))
                conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
                try:
                    conn.execute(f"DELETE FROM memories_vec WHERE rowid IN ({ph})", ids)
                except Exception as ve:
                    logger.warning(f"Failed to delete vector entries for tag '{tag}': {ve}")
                conn.commit()
            return {"success": True, "deleted": len(ids), "sample_keys": keys[:10],
                    "message": f"Deleted {len(ids)} memories with tag '{tag}'"}

        if by == "prefix":
            if not prefix:
                return {"success": False, "error": "prefix parameter required when by='prefix'"}
            with storage._get_conn() as conn:
                escaped = prefix.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                rows = conn.execute(
                    "SELECT id, key FROM memories WHERE key LIKE ? ESCAPE '\\'",
                    (f'{escaped}%',),
                ).fetchall()
                if not rows:
                    return {"success": True, "deleted": 0, "message": f"No memories with key prefix: {prefix}"}
                ids = [r['id'] for r in rows]
                keys = [r['key'] for r in rows]
                ph = ','.join('?' * len(ids))
                conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
                try:
                    conn.execute(f"DELETE FROM memories_vec WHERE rowid IN ({ph})", ids)
                except Exception as ve:
                    logger.warning(f"Failed to delete vector entries for prefix '{prefix}': {ve}")
                conn.commit()
            return {"success": True, "deleted": len(ids), "sample_keys": keys[:10],
                    "message": f"Deleted {len(ids)} memories with key prefix '{prefix}'"}

        return {"success": False, "error": f"Unknown by value: '{by}'. Use key, expired, tag, or prefix."}

    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_update(
    key: str,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    merge: bool = True,
) -> dict:
    """
    Update metadata and/or tags for an existing memory.

    At least one of metadata or tags must be provided.

    Args:
        key      : Memory key to update (required)
        metadata : New metadata dict to set or merge into existing
        tags     : New tag list to set or merge into existing
        merge    : If True (default), merge/append to existing values.
                   If False, replace existing values entirely.

    Returns:
        {"success": bool, "metadata_updated": bool, "tags_updated": bool, "message": ...}
    """
    manager = get_memory_manager()

    try:
        if metadata is None and tags is None:
            return {"success": False, "error": "Provide at least one of: metadata, tags"}

        meta_ok = True
        tags_ok = True

        if metadata is not None:
            meta_ok = manager.update_metadata(key, metadata, merge)
        if tags is not None:
            tags_ok = manager.update_tags(key, tags, merge)

        success = meta_ok and tags_ok
        return {
            "success": success,
            "metadata_updated": bool(metadata is not None and meta_ok),
            "tags_updated": bool(tags is not None and tags_ok),
            "message": f"Memory '{key}' {'updated' if success else 'not found or partially updated'}",
        }
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_list(
    limit: int = 100,
    offset: int = 0,
    tags: list[str] | None = None,
) -> dict:
    """
    List memories with pagination, optionally filtered by tags.

    When tags is empty/None, returns all memories (paginated).
    When tags is non-empty, returns only memories that have ANY of the given tags.

    Args:
        limit  : Maximum number of items to return (1-10000, default 100)
        offset : Number of items to skip for pagination (default 0, ignored when tags provided)
        tags   : Optional list of tags; non-empty activates tag-based filtering

    Returns:
        {"success": True, "total": N, "count": N, "memories": [...]}
        Each memory entry: {key, content (truncated to 200 chars), metadata, tags, updated_at}
    """
    manager = get_memory_manager()

    try:
        if not 1 <= limit <= 10000:
            return {"success": False, "error": "limit must be between 1 and 10000"}
        if offset < 0:
            return {"success": False, "error": "offset must be >= 0"}

        if tags:
            items = manager.list_by_tags(tags, limit)
            total = len(items)
        else:
            items = manager.list(limit=limit, offset=offset)
            total = manager.count()

        return {
            "success": True,
            "total": total,
            "count": len(items),
            "offset": offset,
            "memories": [
                {
                    "key": item.key,
                    "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                    "metadata": item.metadata,
                    "tags": item.tags,
                    "updated_at": item.updated_at,
                }
                for item in items
            ],
        }
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_search(
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
    threshold: float = 0.0,
    metadata_filter: dict[str, Any] | None = None,
    tags_filter: list[str] | None = None,
    vector_weight: float = 0.5,
    text_weight: float = 0.5,
    context_window: int = 0,
) -> dict:
    """
    Search memories using semantic, text, or hybrid search.

    Args:
        query         : Search query (supports Chinese + English)
        top_k         : Maximum number of results (1-1000, default 10)
        mode          : Search mode — one of:
                          "hybrid" (default) — Combines vector + BM25 via RRF for best accuracy
                          "vector"           — Pure semantic/embedding similarity search
                          "text"             — Full-text keyword search (fast, no embeddings)
        threshold     : Minimum similarity score 0-1 (default 0.0); used in hybrid/vector modes
        metadata_filter : Filter results by exact metadata field values
        tags_filter   : Filter results to memories that have any of the given tags
        vector_weight : Weight for vector results in hybrid mode (0-1, default 0.5)
        text_weight   : Weight for text results in hybrid mode (0-1, default 0.5)
        context_window: Number of adjacent chunks to include around each hit (default 0).
                        When > 0, if a result has chunk_index metadata (from file import),
                        its neighboring chunks are automatically fetched and merged into
                        the result content, so the LLM gets full context in one search.
                        Example: context_window=1 returns chunk_004+chunk_005+chunk_006
                        when chunk_005 is the hit.

    Returns:
        hybrid/vector → {"success": True, "mode": ..., "count": N, "results": [{key, content, score, metadata, tags}]}
        text          → {"success": True, "mode": "text", "count": N, "results": [{key, content, metadata, tags}]}
    """
    manager = get_memory_manager()

    try:
        if not 1 <= top_k <= 1000:
            return {"success": False, "error": "top_k must be between 1 and 1000"}
        if not 0.0 <= threshold <= 1.0:
            return {"success": False, "error": "threshold must be between 0.0 and 1.0"}
        if mode not in ("hybrid", "vector", "text"):
            return {"success": False, "error": "mode must be 'hybrid', 'vector', or 'text'"}
        if context_window < 0 or context_window > 10:
            return {"success": False, "error": "context_window must be between 0 and 10"}

        if mode == "text":
            items = manager.search_text(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                tags_filter=tags_filter,
            )
            results_list = [
                {"key": item.key, "content": item.content,
                 "metadata": item.metadata, "tags": item.tags}
                for item in items
            ]
            if context_window > 0:
                results_list = _expand_context_window(manager, results_list, context_window)
            return {
                "success": True,
                "mode": "text",
                "count": len(results_list),
                "results": results_list,
            }

        if mode == "vector":
            results = manager.search(
                query=query,
                top_k=top_k,
                threshold=threshold,
                metadata_filter=metadata_filter,
                tags_filter=tags_filter,
            )
            results_list = [
                {"key": r.key, "content": r.content, "score": round(r.score, 4),
                 "metadata": r.metadata, "tags": r.tags}
                for r in results
            ]
            if context_window > 0:
                results_list = _expand_context_window(manager, results_list, context_window)
            return {
                "success": True,
                "mode": "vector",
                "count": len(results_list),
                "results": results_list,
            }

        # hybrid (default)
        if not 0.0 <= vector_weight <= 1.0:
            return {"success": False, "error": "vector_weight must be between 0.0 and 1.0"}
        if not 0.0 <= text_weight <= 1.0:
            return {"success": False, "error": "text_weight must be between 0.0 and 1.0"}

        results = manager.search_hybrid(
            query=query,
            top_k=top_k,
            threshold=threshold,
            metadata_filter=metadata_filter,
            tags_filter=tags_filter,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )
        results_list = [
            {"key": r.key, "content": r.content, "score": round(r.score, 4),
             "metadata": r.metadata, "tags": r.tags}
            for r in results
        ]
        if context_window > 0:
            results_list = _expand_context_window(manager, results_list, context_window)
        return {
            "success": True,
            "mode": "hybrid",
            "count": len(results_list),
            "results": results_list,
        }

    except Exception as e:
        logger.error(f"Failed to search: {e}")
        return {"success": False, "error": str(e)}


def _expand_context_window(
    manager,
    results: list[dict],
    window: int,
) -> list[dict]:
    """
    Expand each search result with adjacent chunks.

    For results that have chunk_index metadata (from file import),
    fetch neighboring chunks by key pattern and merge their content
    into the result. Non-chunk results are returned unchanged.
    """
    if window <= 0:
        return results

    expanded = []
    seen_keys = set()

    for result in results:
        key = result["key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)

        meta = result.get("metadata") or {}
        chunk_index = meta.get("chunk_index")
        total_chunks = meta.get("total_chunks")

        if chunk_index is None or total_chunks is None:
            expanded.append(result)
            continue

        # Derive the key prefix: "file_xxx_importid_chunk_005" → "file_xxx_importid_chunk_"
        chunk_suffix = f"_chunk_{chunk_index:03d}"
        if not key.endswith(chunk_suffix):
            expanded.append(result)
            continue
        key_prefix = key[: -len(chunk_suffix)] + "_chunk_"

        # Determine which chunk indices to fetch
        start_idx = max(0, chunk_index - window)
        end_idx = min(total_chunks - 1, chunk_index + window)

        # Fetch neighboring chunks
        parts = []
        neighbor_keys = set()
        for idx in range(start_idx, end_idx + 1):
            neighbor_key = f"{key_prefix}{idx:03d}"
            neighbor_keys.add(neighbor_key)
            if neighbor_key == key:
                parts.append((idx, result["content"]))
            else:
                neighbor = manager.get(neighbor_key)
                if neighbor:
                    parts.append((idx, neighbor.content))

        # Sort by index and merge
        parts.sort(key=lambda x: x[0])
        merged_content = "\n\n".join(p[1] for p in parts)

        # Mark neighbor keys as seen to avoid duplicates
        seen_keys.update(neighbor_keys)

        expanded_result = {
            **result,
            "content": merged_content,
            "context_expanded": True,
            "context_chunks": [start_idx, end_idx],
        }
        expanded.append(expanded_result)

    return expanded


# ============================================================================
# Utility Operations
# ============================================================================

@mcp.tool()
def memory_similarity(text1: str, text2: str) -> dict:
    """
    Calculate semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with similarity score (0-1)
    """
    manager = get_memory_manager()
    
    try:
        score = manager.similarity(text1, text2)
        return {
            "success": True,
            "similarity": round(score, 4),
            "interpretation": (
                "Very similar" if score > 0.8 else
                "Similar" if score > 0.6 else
                "Somewhat related" if score > 0.4 else
                "Different" if score > 0.2 else
                "Very different"
            ),
        }
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_stats(
    cleanup: bool = False,
    cleanup_min_length: int = 50,
    cleanup_dry_run: bool = True,
) -> dict:
    """
    Get memory system statistics, and optionally trigger a garbage cleanup.

    Stats mode (cleanup=False, default):
        Returns comprehensive system statistics (count, DB size, tag distribution, etc.).

    Cleanup mode (cleanup=True):
        Finds (and optionally deletes) memories whose content is shorter than
        cleanup_min_length characters. These are typically junk fragments from
        bad file imports (e.g. content like `)`  or  `])`  that pollute search).

        cleanup_min_length : Minimum content length to keep (default 50 chars)
        cleanup_dry_run    : If True (default), only report what would be deleted
                             without actually deleting. Set to False to actually delete.

    Returns:
        Stats mode   → {"success": True, ...stats...}
        Cleanup mode → {"success": True, "dry_run": bool, "found"/"deleted": N,
                        "samples": [...], "message": ...}
    """
    manager = get_memory_manager()

    try:
        if cleanup:
            storage = manager.storage
            with storage._get_conn() as conn:
                rows = conn.execute("""
                    SELECT id, key, content, LENGTH(content) as content_len
                    FROM memories
                    WHERE LENGTH(content) < ?
                    ORDER BY content_len ASC
                """, (cleanup_min_length,)).fetchall()

                if not rows:
                    return {
                        "success": True,
                        "found": 0,
                        "message": f"No memories with content shorter than {cleanup_min_length} chars",
                    }

                garbage_ids = [r['id'] for r in rows]
                samples = [
                    {"key": r['key'], "content": r['content'], "length": r['content_len']}
                    for r in rows[:20]
                ]

                if cleanup_dry_run:
                    return {
                        "success": True,
                        "dry_run": True,
                        "found": len(rows),
                        "samples": samples,
                        "message": (
                            f"Found {len(rows)} garbage memories (< {cleanup_min_length} chars). "
                            "Set cleanup_dry_run=False to delete them."
                        ),
                    }

                ph = ','.join('?' * len(garbage_ids))
                conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", garbage_ids)
                try:
                    conn.execute(f"DELETE FROM memories_vec WHERE rowid IN ({ph})", garbage_ids)
                except Exception as ve:
                    logger.warning(f"Failed to delete vector entries during cleanup: {ve}")
                conn.commit()
                remaining = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            return {
                "success": True,
                "dry_run": False,
                "deleted": len(garbage_ids),
                "remaining": remaining,
                "samples": samples,
                "message": f"Deleted {len(garbage_ids)} garbage memories. {remaining} remaining.",
            }

        stats = manager.get_stats()
        return {"success": True, **stats}

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def memory_import(
    action: str = "json",
    memories: list[dict] | None = None,
    file_path: str = "",
    chunk_size: int = 2000,
    overlap: int = 200,
    min_chunk_size: int = 50,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """
    Import/export memories. Controlled by the action parameter.

    action="json" (default) — Import from a list of memory dicts:
        memories : List of memory dicts (as returned by action="export").
                   Each dict must have at least a "content" key; "key",
                   "metadata", "tags" are optional.

    action="export" — Export all memories as JSON-serialisable data:
        No additional parameters needed.
        Returns {"success": True, "count": N, "memories": [...]}

    action="file" — Import a file into memory with intelligent chunking:
        file_path      : Absolute path to the file to import (required)
        chunk_size     : Max characters per chunk (default 2000)
        overlap        : Overlap between text chunks for context (default 200)
        min_chunk_size : Minimum chunk length; shorter chunks are merged (default 50)
        tags           : Optional tags applied to all generated chunks
        metadata       : Optional metadata applied to all generated chunks

        For code files (.py .js .ts .cpp .go .rs …) the file is split on
        function/class boundaries; for prose/text, on paragraph/sentence breaks.

    Returns:
        action=json   → {"success": True, "imported_count": N, "message": ...}
        action=export → {"success": True, "count": N, "memories": [...]}
        action=file   → {"success": True, "chunks": N, "keys": [...], ...}
    """
    import os
    from pathlib import Path

    manager = get_memory_manager()

    try:
        if action == "export":
            data = manager.export_all()
            return {"success": True, "count": len(data), "memories": data}

        if action == "json":
            if not memories:
                return {"success": False, "error": "memories list is required for action='json'"}
            count = manager.import_batch(memories)
            return {"success": True, "imported_count": count,
                    "message": f"Imported {count} memories"}

        if action == "file":
            # Validate chunk params
            if chunk_size < 100:
                return {"success": False, "error": "chunk_size must be at least 100"}
            if overlap < 0 or overlap >= chunk_size:
                return {"success": False, "error": "overlap must be >= 0 and < chunk_size"}
            if min_chunk_size < 0 or min_chunk_size > chunk_size:
                return {"success": False, "error": "min_chunk_size must be >= 0 and <= chunk_size"}

            raw_parts = Path(file_path).parts
            if '..' in raw_parts:
                return {"success": False, "error": "Path traversal not allowed: '..' in path"}
            path = Path(file_path).resolve()
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            if not path.is_file():
                return {"success": False, "error": f"Not a file: {file_path}"}

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

            if not content.strip():
                return {"success": False, "error": "File is empty"}

            file_name = path.stem
            file_ext = path.suffix.lower()
            base_metadata = {
                "source_file": str(path),
                "file_name": path.name,
                "file_type": file_ext,
                **(metadata or {}),
            }
            base_tags = list(tags or []) + ["imported_file"]
            import_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

            if len(content) <= chunk_size:
                key = manager.save(
                    content=content,
                    key=f"file_{file_name}_{import_id}",
                    metadata=base_metadata,
                    tags=base_tags,
                )
                return {"success": True, "chunks": 1, "keys": [key],
                        "message": f"Imported file as single memory: {key}"}

            CODE_EXTENSIONS = {
                ".py", ".cpp", ".c", ".h", ".hpp", ".js", ".ts", ".jsx", ".tsx",
                ".rs", ".go", ".java", ".cs", ".lua", ".rb", ".php", ".swift",
                ".kt", ".scala", ".zig",
            }

            if file_ext in CODE_EXTENSIONS:
                raw_chunks = _chunk_code(content, file_ext, chunk_size)
            else:
                raw_chunks = _chunk_text(content, chunk_size, overlap)

            merged_chunks = _merge_small_chunks(raw_chunks, min_chunk_size, chunk_size)

            items_list = []
            for i, chunk_content in enumerate(merged_chunks):
                items_list.append({
                    "content": chunk_content,
                    "key": f"file_{file_name}_{import_id}_chunk_{i:03d}",
                    "metadata": {**base_metadata, "chunk_index": i,
                                 "total_chunks": len(merged_chunks)},
                    "tags": base_tags + [f"chunk_{i}"],
                })

            if not items_list:
                return {"success": False, "error": "No valid chunks generated from file"}

            # Use Late Chunking if supported (produces context-aware embeddings)
            keys = manager.save_batch_late_chunking(content, items_list)
            return {
                "success": True,
                "chunks": len(keys),
                "keys": keys,
                "total_chars": len(content),
                "avg_chunk_size": len(content) // len(keys),
                "message": f"Imported file as {len(keys)} chunks (avg {len(content) // len(keys)} chars)",
            }

        return {"success": False, "error": f"Unknown action: '{action}'. Use 'json', 'export', or 'file'."}

    except Exception as e:
        logger.error(f"Failed to import/export: {e}")
        return {"success": False, "error": str(e)}


def _chunk_code(content: str, file_ext: str, chunk_size: int) -> list[str]:
    """
    Split code files on function/class/struct boundaries.
    
    Keeps complete logical blocks together instead of blindly
    splitting by character count, which destroys code semantics.
    """
    import re
    
    # Regex patterns for top-level definitions per language
    PATTERNS = {
        ".py": r'^(?=(?:class |def |async def ))',
        ".cpp": r'^(?=(?:class |struct |namespace |(?:[\w:*&<> ]+\s+)?[\w:]+\s*\([^;]*$))',
        ".c": r'^(?=(?:(?:static |extern |inline )?(?:[\w*]+ )+\w+\s*\())',
        ".h": r'^(?=(?:class |struct |namespace |#pragma|#ifndef|#define))',
        ".hpp": r'^(?=(?:class |struct |namespace |template))',
        ".js": r'^(?=(?:function |class |const \w+ = |export ))',
        ".ts": r'^(?=(?:function |class |interface |type |const \w+ = |export ))',
        ".jsx": r'^(?=(?:function |class |const \w+ = |export ))',
        ".tsx": r'^(?=(?:function |class |interface |type |const \w+ = |export ))',
        ".rs": r'^(?=(?:fn |pub fn |impl |struct |enum |trait |mod ))',
        ".go": r'^(?=(?:func |type |package ))',
        ".java": r'^(?=(?:public |private |protected |class |interface |enum ))',
        ".cs": r'^(?=(?:public |private |protected |internal |class |struct |interface |enum |namespace ))',
        ".lua": r'^(?=(?:function |local function ))',
        ".rb": r'^(?=(?:def |class |module ))',
        ".php": r'^(?=(?:function |class |interface |trait |namespace ))',
        ".swift": r'^(?=(?:func |class |struct |enum |protocol |extension ))',
        ".kt": r'^(?=(?:fun |class |interface |object |enum ))',
        ".scala": r'^(?=(?:def |class |trait |object |case class ))',
        ".zig": r'^(?=(?:pub fn |fn |const |pub const ))',
    }
    
    pattern = PATTERNS.get(file_ext, r'^(?=\S)')
    
    # Split into blocks at top-level definitions
    blocks = re.split(pattern, content, flags=re.MULTILINE)
    blocks = [b for b in blocks if b.strip()]
    
    if not blocks:
        # Fallback to text chunking if regex didn't split
        return _chunk_text(content, chunk_size, chunk_size // 5)
    
    # Merge blocks that are too small, split blocks that are too large
    chunks = []
    current = ""
    
    for block in blocks:
        if len(current) + len(block) <= chunk_size:
            current += block
        else:
            if current.strip():
                chunks.append(current.strip())
            # If single block exceeds chunk_size, split it by lines
            if len(block) > chunk_size:
                sub_chunks = _split_large_block(block, chunk_size)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = block
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def _split_large_block(block: str, chunk_size: int) -> list[str]:
    """Split an oversized code block by line groups."""
    lines = block.split("\n")
    chunks = []
    current_lines = []
    current_len = 0
    
    for line in lines:
        if current_len + len(line) + 1 > chunk_size and current_lines:
            chunks.append("\n".join(current_lines).strip())
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line) + 1
    
    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(text)
    
    return chunks


def _chunk_text(content: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text content on paragraph/sentence boundaries.
    
    Improved algorithm that prevents tiny fragment generation:
    - Prefers paragraph breaks (\n\n), then line breaks (\n), then sentence ends
    - Guarantees forward progress of at least chunk_size // 2 per iteration
    - Never produces chunks shorter than min stride
    """
    chunks = []
    start = 0
    min_stride = max(chunk_size // 4, 100)  # Minimum forward progress
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # Try to break at natural boundaries (only if not at end of content)
        if end < len(content):
            best_break = -1
            
            # Priority 1: paragraph break (\n\n)
            pos = content.rfind("\n\n", start + min_stride, end)
            if pos > start + min_stride:
                best_break = pos + 2  # Include the double newline
            
            # Priority 2: line break (\n)
            if best_break == -1:
                pos = content.rfind("\n", start + min_stride, end)
                if pos > start + min_stride:
                    best_break = pos + 1
            
            # Priority 3: sentence end
            if best_break == -1:
                for sep in [". ", "。", "! ", "? ", ";\n", "；\n"]:
                    pos = content.rfind(sep, start + min_stride, end)
                    if pos > start + min_stride:
                        best_break = pos + len(sep)
                        break
            
            if best_break > start + min_stride:
                end = best_break
        
        chunk_content = content[start:end].strip()
        if chunk_content:
            chunks.append(chunk_content)
        
        # Guaranteed forward progress: move at least (end - overlap) or min_stride
        next_start = end - overlap
        start = max(next_start, start + min_stride)
    
    return chunks


def _merge_small_chunks(chunks: list[str], min_size: int, max_size: int) -> list[str]:
    """
    Merge chunks that are too small with their neighbors.
    
    This is the key defense against garbage fragments.
    Any chunk shorter than min_size gets merged into the previous chunk.
    """
    if not chunks:
        return []
    
    merged = []
    buffer = ""
    
    for chunk in chunks:
        if len(chunk) < min_size:
            # Too small: accumulate into buffer
            buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        else:
            # Flush buffer by prepending to this chunk
            if buffer:
                combined = buffer + "\n\n" + chunk
                if len(combined) <= max_size:
                    chunk = combined
                else:
                    # Buffer is big enough on its own now, emit separately
                    if len(buffer) >= min_size:
                        merged.append(buffer)
                    else:
                        # Append to previous chunk if possible
                        if merged:
                            merged[-1] = merged[-1] + "\n\n" + buffer
                buffer = ""
            merged.append(chunk)
    
    # Handle remaining buffer
    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        elif len(buffer.strip()) >= min_size:
            merged.append(buffer)
    
    return merged


# ============================================================================
# Server Entry Point
# ============================================================================

def run_server():
    """Run the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting Ultra-Light Memory MCP Server...")
    
    # Warmup embedding model
    try:
        get_memory_manager().warmup()
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
    
    # Run server
    mcp.run()


if __name__ == "__main__":
    run_server()
