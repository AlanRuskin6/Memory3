"""
Advanced Embedding Module with Multi-Model Support.

Supports:
- BGE-M3: State-of-the-art multilingual embedding (best quality)
- BGE-Small-ZH: Lightweight Chinese embedding (balanced)
- All-MiniLM: Ultra-light English fallback

Features:
- Matryoshka dimension reduction
- Automatic GPU/CPU/MPS detection
- Batch encoding with progress
- Thread-safe model loading
- Model caching
"""

import logging
import threading
from typing import Optional, Literal
import numpy as np

logger = logging.getLogger(__name__)

# Model configurations - curated for best quality/size tradeoffs
MODEL_CONFIGS = {
    "jina-v3": {
        "name": "jinaai/jina-embeddings-v3",
        "max_dim": 1024,
        "supported_dims": [1024, 768, 512, 384, 256],
        "description": "Long-context 8192-token, Late Chunking support, multilingual",
        "size_mb": 570,
        "max_seq_length": 8192,
        "late_chunking": True,
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "max_dim": 1024,
        "supported_dims": [1024, 768, 512, 384, 256],
        "description": "State-of-the-art multilingual (100+ languages)",
        "size_mb": 2200,
    },
    "bge-small-zh": {
        "name": "BAAI/bge-small-zh-v1.5",
        "max_dim": 512,
        "supported_dims": [512, 384, 256, 128],
        "description": "Lightweight Chinese embedding",
        "size_mb": 95,
    },
    "bge-small-en": {
        "name": "BAAI/bge-small-en-v1.5",
        "max_dim": 384,
        "supported_dims": [384, 256, 128],
        "description": "Lightweight English embedding",
        "size_mb": 130,
    },
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_dim": 384,
        "supported_dims": [384, 256, 128],
        "description": "Ultra-light English fallback",
        "size_mb": 90,
    },
}

# Default settings
DEFAULT_MODEL_KEY = "bge-m3"
DEFAULT_DIM = 1024
FALLBACK_MODEL_KEY = "bge-small-zh"

# Models that support late chunking (long-context token-level encoding)
LATE_CHUNKING_MODELS = {k for k, v in MODEL_CONFIGS.items() if v.get("late_chunking")}


class EmbeddingEngine:
    """
    High-performance embedding engine with multi-model support.
    
    Features:
    - Lazy model loading with thread safety
    - Matryoshka dimension reduction
    - Automatic device selection (CUDA/MPS/CPU)
    - Batch encoding with progress bars
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_key: Optional[str] = None,
        dimension: int = DEFAULT_DIM,
        device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
        use_gpu: bool = True,
        max_seq_length: int = 512,
    ):
        """
        Initialize embedding engine.
        
        Args:
            model_name: Direct HuggingFace model name (overrides model_key)
            model_key: Preset model key (bge-m3, bge-small-zh, minilm, etc.)
            dimension: Output embedding dimension (Matryoshka truncation)
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
            use_gpu: Whether to attempt GPU acceleration
            max_seq_length: Maximum sequence length for encoding
        """
        # Resolve model
        if model_name:
            self.model_name = model_name
            self.model_key = "custom"
        else:
            self.model_key = model_key or DEFAULT_MODEL_KEY
            if self.model_key not in MODEL_CONFIGS:
                logger.warning(f"Unknown model key: {self.model_key}, using {DEFAULT_MODEL_KEY}")
                self.model_key = DEFAULT_MODEL_KEY
            self.model_name = MODEL_CONFIGS[self.model_key]["name"]
        
        self.dimension = dimension
        self.device = device
        self.use_gpu = use_gpu
        self.max_seq_length = max_seq_length
        self._model = None
        self._lock = threading.Lock()  # C4: initialized at __init__ time, not lazily
        
        # Validate dimension
        if self.model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[self.model_key]
            if dimension > config["max_dim"]:
                logger.warning(
                    f"Dimension {dimension} > model max {config['max_dim']}. "
                    f"Using {config['max_dim']}"
                )
                self.dimension = config["max_dim"]
    
    @property
    def model(self):
        """Lazy load the model (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:  # Double-check locking
                    self._load_model()
        return self._model
    
    def _resolve_device(self) -> str:
        """Resolve the device to use."""
        if self.device != "auto":
            return self.device
        
        if not self.use_gpu:
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
        
        device = self._resolve_device()
        logger.info(f"Loading model {self.model_name} on {device}...")
        
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )
            
            # Set max sequence length
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.max_seq_length
            
            logger.info(f"✓ Loaded {self.model_name} (dim={self._model.get_sentence_embedding_dimension()})")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info(f"Falling back to {MODEL_CONFIGS[FALLBACK_MODEL_KEY]['name']}")
            
            self._model = SentenceTransformer(
                MODEL_CONFIGS[FALLBACK_MODEL_KEY]["name"],
                device=device,
            )
            self.model_name = MODEL_CONFIGS[FALLBACK_MODEL_KEY]["name"]
            self.model_key = FALLBACK_MODEL_KEY
    
    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar for batch encoding
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings, shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # Encode with model
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        
        # Matryoshka dimension truncation
        if embeddings.shape[1] > self.dimension:
            embeddings = embeddings[:, :self.dimension]
            # Re-normalize after truncation
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-10)
        
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text, return 1D array."""
        return self.encode([text], normalize=normalize)[0]
    
    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode a batch of texts with progress bar."""
        return self.encode(
            texts,
            normalize=normalize,
            show_progress=show_progress,
            batch_size=batch_size,
        )
    
    @property
    def embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.dimension
    
    @property
    def native_dim(self) -> int:
        """Get the model's native embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts. Returns value in [-1, 1]."""
        emb1 = self.encode_single(text1, normalize=True)
        emb2 = self.encode_single(text2, normalize=True)
        # encode_single(normalize=True) guarantees unit vectors → dot product = cosine sim
        return float(np.dot(emb1, emb2))
    
    @property
    def supports_late_chunking(self) -> bool:
        """Check if the current model supports late chunking."""
        return self.model_key in LATE_CHUNKING_MODELS

    def encode_late_chunks(
        self,
        full_text: str,
        chunk_texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Late Chunking: encode the full document once through the transformer,
        then pool token embeddings per chunk boundary.

        This produces chunk embeddings that carry full-document context,
        dramatically improving retrieval for chunks with pronouns,
        abbreviations, or implicit references.

        Falls back to standard `encode()` if the model does not support
        late chunking or if the full text exceeds the model's context window.

        Args:
            full_text:   The entire document text.
            chunk_texts: List of chunk strings (must be contiguous substrings
                         of full_text in order, covering it fully or partially).
            normalize:   L2-normalize the resulting embeddings.

        Returns:
            np.ndarray of shape (len(chunk_texts), self.dimension)
        """
        if not self.supports_late_chunking:
            logger.debug("Model does not support late chunking, falling back to standard encode")
            return self.encode(chunk_texts, normalize=normalize)

        try:
            model = self.model
            tokenizer = model.tokenizer

            # Tokenize the full document
            full_encoding = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

            # Check if text was truncated
            token_count = full_encoding["input_ids"].shape[1]
            if token_count >= self.max_seq_length:
                logger.warning(
                    f"Document ({token_count} tokens) exceeds model context "
                    f"({self.max_seq_length}). Falling back to standard encode."
                )
                return self.encode(chunk_texts, normalize=normalize)

            # Move to model device
            import torch
            device = model.device
            full_encoding = {k: v.to(device) for k, v in full_encoding.items()}

            # Forward pass — get token-level embeddings from the transformer
            with torch.no_grad():
                outputs = model[0].auto_model(**full_encoding)
                # last_hidden_state: (1, seq_len, hidden_dim)
                token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

            # Find chunk boundaries via tokenizer offsets
            full_tokens = tokenizer(
                full_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )
            offsets = full_tokens["offset_mapping"]  # list of (start, end) char positions

            # Map each chunk to token spans
            chunk_embeddings = []
            for chunk_text in chunk_texts:
                # Find the chunk's character span in full_text
                chunk_start = full_text.find(chunk_text)
                if chunk_start == -1:
                    # Chunk not found as substring, fall back to individual encode
                    logger.debug("Chunk not found in full_text, encoding individually")
                    chunk_embeddings.append(self.encode_single(chunk_text, normalize=False))
                    continue
                chunk_end = chunk_start + len(chunk_text)

                # Find token indices that overlap with the chunk span
                token_indices = []
                for t_idx, (t_start, t_end) in enumerate(offsets):
                    if t_end == 0 and t_start == 0:
                        continue  # special token
                    if t_start < chunk_end and t_end > chunk_start:
                        token_indices.append(t_idx)

                if not token_indices:
                    chunk_embeddings.append(self.encode_single(chunk_text, normalize=False))
                    continue

                # Mean pool the token embeddings for this chunk
                chunk_token_embs = token_embeddings[token_indices]  # (n_tokens, hidden_dim)
                pooled = chunk_token_embs.mean(dim=0)  # (hidden_dim,)
                chunk_embeddings.append(pooled.cpu().numpy())

            embeddings = np.array(chunk_embeddings, dtype=np.float32)

            # Matryoshka dimension truncation
            if embeddings.shape[1] > self.dimension:
                embeddings = embeddings[:, :self.dimension]

            # L2 normalize
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-10)

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.warning(f"Late chunking failed: {e}. Falling back to standard encode.")
            return self.encode(chunk_texts, normalize=normalize)

    def warmup(self):
        """Warmup the model with a dummy encoding."""
        logger.info("Warming up embedding model...")
        self.encode("warmup")
        logger.info("✓ Model ready")
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "output_dimension": self.dimension,
            "native_dimension": self.native_dim if self._model else None,
            "device": self._resolve_device(),
            "max_seq_length": self.max_seq_length,
        }


# Global engine cache
_engines: dict[str, EmbeddingEngine] = {}


def get_engine(
    model_key: Optional[str] = None,
    model_name: Optional[str] = None,
    dimension: int = DEFAULT_DIM,
    use_gpu: bool = True,
    **kwargs,
) -> EmbeddingEngine:
    """
    Get or create an embedding engine.
    
    Engines are cached by (model_key, dimension) for reuse.
    """
    key = model_key or DEFAULT_MODEL_KEY
    cache_key = f"{key}:{dimension}"
    
    if cache_key not in _engines:
        _engines[cache_key] = EmbeddingEngine(
            model_key=model_key,
            model_name=model_name,
            dimension=dimension,
            use_gpu=use_gpu,
            **kwargs,
        )
    
    return _engines[cache_key]


def encode(texts: list[str] | str, normalize: bool = True) -> np.ndarray:
    """Convenience function to encode texts using default engine."""
    return get_engine().encode(texts, normalize=normalize)


def list_models() -> dict:
    """List available model presets."""
    return {
        key: {
            "name": config["name"],
            "description": config["description"],
            "dimensions": config["supported_dims"],
            "size_mb": config["size_mb"],
        }
        for key, config in MODEL_CONFIGS.items()
    }
