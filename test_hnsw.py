"""Quick test for HNSW integration."""
import sys
print(f"Python: {sys.version}")

try:
    import hnswlib
    print("✓ hnswlib imported successfully")
except ImportError as e:
    print(f"✗ hnswlib import failed: {e}")
    sys.exit(1)

try:
    from ultra_light_memory.hnsw_index import HNSWIndex, HNSW_AVAILABLE
    print(f"✓ HNSWIndex imported, HNSW_AVAILABLE={HNSW_AVAILABLE}")
except ImportError as e:
    print(f"✗ HNSWIndex import failed: {e}")
    sys.exit(1)

import numpy as np

# Test HNSW index
print("\n--- Testing HNSWIndex ---")
idx = HNSWIndex(dimension=384, max_elements=1000)
print("✓ HNSWIndex created")

# Add vectors
for i in range(100):
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    idx.add(i, vec)
print("✓ Added 100 vectors")

# Search
query = np.random.randn(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = idx.search(query, k=5)
print(f"✓ Search returned {len(results)} results")
print(f"  Top 3: {results[:3]}")

# Stats
stats = idx.get_stats()
print(f"\n--- Stats ---")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n🎉 HNSW TEST PASSED!")
