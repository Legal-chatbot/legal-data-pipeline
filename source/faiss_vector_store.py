"""FAISS vector store with a JSON metadata sidecar for legal chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from source.embedding_service import LegalChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorStoreConfig:
    dimension: int
    top_k: int = 5
    metric: str = "cosine"
    index_type: str = "flat"
    hnsw_m: int = 32

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be greater than zero")
        if self.top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if self.metric not in {"cosine", "ip", "l2"}:
            raise ValueError("metric must be one of: cosine, ip, l2")
        if self.index_type not in {"flat", "hnsw"}:
            raise ValueError("index_type must be one of: flat, hnsw")
        if self.hnsw_m < 1:
            raise ValueError("hnsw_m must be greater than zero")


@dataclass(frozen=True)
class SearchResult:
    vector_id: int
    score: float
    chunk: LegalChunk


class FaissVectorStore:
    """Store vectors in FAISS and legal chunk metadata in a sidecar mapping."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self.index = None
        self._metadata: dict[int, LegalChunk] = {}
        self._next_vector_id = 0

    @property
    def size(self) -> int:
        return len(self._metadata)

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def create_index(self) -> None:
        import faiss

        faiss_metric = (
            faiss.METRIC_L2
            if self.config.metric == "l2"
            else faiss.METRIC_INNER_PRODUCT
        )
        if self.config.index_type == "flat":
            base_index = faiss.IndexFlat(self.dimension, faiss_metric)
        else:
            base_index = faiss.IndexHNSWFlat(
                self.dimension,
                self.config.hnsw_m,
                faiss_metric,
            )
        self.index = faiss.IndexIDMap2(base_index)
        self._metadata.clear()
        self._next_vector_id = 0
        logger.info(
            "Created FAISS index: type=%s metric=%s dimension=%d",
            self.config.index_type,
            self.config.metric,
            self.dimension,
        )

    def _require_index(self) -> Any:
        if self.index is None:
            raise RuntimeError("index has not been created or loaded")
        return self.index

    def _prepare_vectors(self, vectors: np.ndarray, *, normalize: bool) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            actual = values.shape[1] if values.ndim == 2 else None
            raise ValueError(
                f"expected vectors with shape (n, {self.dimension}), got {values.shape}; "
                f"actual dimension={actual}"
            )
        if not np.isfinite(values).all():
            raise ValueError("vectors must contain only finite values")
        values = np.ascontiguousarray(values, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        return values

    def add_vectors(
        self,
        vectors: np.ndarray,
        chunks: Sequence[LegalChunk],
        vector_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """Add vectors and their metadata, returning the FAISS vector IDs."""

        index = self._require_index()
        values = self._prepare_vectors(vectors, normalize=self.config.metric == "cosine")
        if len(values) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        if not len(values):
            return []

        ids = list(vector_ids) if vector_ids is not None else list(
            range(self._next_vector_id, self._next_vector_id + len(values))
        )
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("vector_ids must be unique and match the number of vectors")
        if any(not isinstance(vector_id, (int, np.integer)) for vector_id in ids):
            raise ValueError("vector_ids must contain integers")
        if any(int(vector_id) in self._metadata for vector_id in ids):
            raise ValueError("vector_ids must not already exist in the index")

        ids_array = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(values, ids_array)
        self._metadata.update({int(vector_id): chunk for vector_id, chunk in zip(ids, chunks)})
        self._next_vector_id = max(self._next_vector_id, max(ids_array).item() + 1)
        logger.info("Added %d vectors; total=%d", len(values), self.size)
        return [int(vector_id) for vector_id in ids]

    def delete(self, vector_ids: Sequence[int]) -> int:
        """Delete vectors and metadata by FAISS vector ID."""

        index = self._require_index()
        ids = [int(vector_id) for vector_id in vector_ids]
        if not ids:
            return 0
        removed = int(index.remove_ids(np.asarray(ids, dtype=np.int64)))
        for vector_id in ids:
            self._metadata.pop(vector_id, None)
        logger.info("Deleted %d vectors; total=%d", removed, self.size)
        return removed

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return nearest chunks ordered by descending similarity score."""

        index = self._require_index()
        requested_k = self.config.top_k if top_k is None else top_k
        if requested_k < 1:
            raise ValueError("top_k must be greater than zero")
        if self.size == 0:
            return []

        query = self._prepare_vectors(
            query_embedding,
            normalize=self.config.metric == "cosine",
        )
        distances, ids = index.search(query, min(requested_k, self.size))
        results = []
        for distance, vector_id in zip(distances[0], ids[0]):
            vector_id = int(vector_id)
            if vector_id == -1 or vector_id not in self._metadata:
                continue
            score = float(-distance if self.config.metric == "l2" else distance)
            results.append(
                SearchResult(vector_id=vector_id, score=score, chunk=self._metadata[vector_id])
            )
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path | None = None) -> Path:
        """Save FAISS data and the vector ID to ``LegalChunk`` mapping."""

        import faiss

        index = self._require_index()
        index_path = Path(index_path)
        metadata_path = Path(metadata_path or index_path.with_suffix(".metadata.json"))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        payload = {
            "config": asdict(self.config),
            "next_vector_id": self._next_vector_id,
            "chunks": {str(vector_id): self._chunk_to_dict(chunk) for vector_id, chunk in self._metadata.items()},
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved FAISS index=%s metadata=%s", index_path, metadata_path)
        return index_path

    def load(self, index_path: str | Path, metadata_path: str | Path | None = None) -> None:
        """Load an index and its metadata mapping into this store."""

        import faiss

        index_path = Path(index_path)
        metadata_path = Path(metadata_path or index_path.with_suffix(".metadata.json"))
        loaded_index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded_config = VectorStoreConfig(**payload["config"])
        if loaded_config.dimension != self.dimension:
            raise ValueError("saved index dimension does not match store dimension")
        self.index = loaded_index
        self._metadata = {
            int(vector_id): self._chunk_from_dict(chunk)
            for vector_id, chunk in payload["chunks"].items()
        }
        if self.index.ntotal != len(self._metadata):
            raise ValueError("FAISS index and metadata mapping have different sizes")
        self._next_vector_id = int(payload["next_vector_id"])
        logger.info("Loaded FAISS index=%s; total=%d", index_path, self.size)

    @staticmethod
    def _chunk_to_dict(chunk: LegalChunk) -> dict[str, Any]:
        return {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "metadata": dict(chunk.metadata),
        }

    @staticmethod
    def _chunk_from_dict(value: Mapping[str, Any]) -> LegalChunk:
        return LegalChunk(
            text=value["text"],
            chunk_id=value.get("chunk_id"),
            metadata=value.get("metadata", {}),
        )