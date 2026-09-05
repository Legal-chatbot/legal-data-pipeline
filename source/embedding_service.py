"""Embedding service for legal chunks.

The service deliberately owns only model loading and encoding. Indexing and
storage integrations can depend on this interface without depending on a
particular embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegalChunk:
    """Minimal chunk contract accepted by :class:`EmbeddingService`."""

    text: str
    chunk_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Runtime configuration for one embedding model instance."""

    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    max_seq_length: int = 8192
    device: str | None = None
    normalize_embeddings: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be greater than zero")


class EmbeddingModel(Protocol):
    max_seq_length: int

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any:
        ...


ModelFactory = Callable[[EmbeddingConfig, str], EmbeddingModel]


def detect_device() -> str:
    """Select the best available PyTorch device without hard-coding one."""

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_model_factory(config: EmbeddingConfig, device: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.model_name,
        device=device,
        trust_remote_code=config.trust_remote_code,
    )
    model.max_seq_length = config.max_seq_length
    return model


class EmbeddingService:
    """Encode text, queries, and legal chunks with a cached model backend."""

    _model_cache: dict[tuple[Any, ...], EmbeddingModel] = {}
    _cache_lock = Lock()

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self.device = self.config.device or detect_device()
        self._model_factory = model_factory or _default_model_factory
        self._model: EmbeddingModel | None = None

    @classmethod
    def clear_model_cache(cls) -> None:
        with cls._cache_lock:
            cls._model_cache.clear()

    def _get_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model

        cache_key = (
            self.config.model_name,
            self.device,
            self.config.max_seq_length,
            self.config.trust_remote_code,
            id(self._model_factory),
        )
        with self._cache_lock:
            model = self._model_cache.get(cache_key)
            if model is None:
                logger.info("Loading embedding model: %s", self.config.model_name)
                model = self._model_factory(self.config, self.device)
                self._model_cache[cache_key] = model
                logger.info("Embedding model loaded on device: %s", self.device)
        self._model = model
        return model

    @staticmethod
    def _text_from_chunk(chunk: LegalChunk | Mapping[str, Any] | Any) -> str:
        if isinstance(chunk, Mapping):
            text = chunk.get("text")
        else:
            text = getattr(chunk, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each LegalChunk must contain non-empty text")
        return text

    def embed_text(self, text: str) -> np.ndarray:
        """Return one normalized float32 embedding vector for ``text``."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        return self.embed_chunks([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query using the same model contract as documents."""

        return self.embed_text(query)

    def embed_chunks(
        self,
        chunks: Sequence[LegalChunk | Mapping[str, Any] | Any] | Sequence[str],
    ) -> np.ndarray:
        """Return an ``(n_chunks, embedding_dim)`` matrix for chunks or texts."""

        if not chunks:
            logger.info("Embedding chunks: number of chunks=0")
            return np.empty((0, 0), dtype=np.float32)

        texts = [
            item if isinstance(item, str) else self._text_from_chunk(item)
            for item in chunks
        ]
        if any(not text.strip() for text in texts):
            raise ValueError("texts must be non-empty strings")

        model = self._get_model()
        logger.info(
            "Embedding chunks: number of chunks=%d, batch size=%d",
            len(texts),
            self.config.batch_size,
        )
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        result = np.asarray(embeddings, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = np.divide(
                result,
                norms,
                out=np.zeros_like(result),
                where=norms > 0,
            )
        logger.info("Embedding dimension: %d", result.shape[1])
        return result