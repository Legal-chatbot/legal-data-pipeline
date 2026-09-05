import numpy as np
import pytest

from source.embedding_service import LegalChunk
from source.faiss_vector_store import FaissVectorStore, VectorStoreConfig


def chunks(count):
    return [LegalChunk(f"Điều {index}", chunk_id=str(index)) for index in range(count)]


def make_store(dimension=3, **kwargs):
    store = FaissVectorStore(VectorStoreConfig(dimension=dimension, **kwargs))
    store.create_index()
    return store


def test_index_creation_and_add_vectors():
    store = make_store()

    ids = store.add_vectors(np.eye(3, dtype=np.float32), chunks(3))

    assert ids == [0, 1, 2]
    assert store.size == 3
    assert store.index.ntotal == 3


def test_save_load_preserves_metadata_and_search(tmp_path):
    store = make_store()
    store.add_vectors(np.eye(3, dtype=np.float32), chunks(3))
    index_path = tmp_path / "legal.faiss"

    store.save(index_path)
    loaded = make_store()
    loaded.load(index_path)
    results = loaded.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)

    assert results[0].vector_id == 1
    assert results[0].chunk.chunk_id == "1"
    assert results[0].score == pytest.approx(1.0)


def test_search_returns_relevant_chunks_and_respects_top_k():
    store = make_store()
    store.add_vectors(
        np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.2, 0]], dtype=np.float32),
        chunks(3),
    )

    results = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)

    assert [result.vector_id for result in results] == [0, 2]
    assert results[0].chunk.text == "Điều 0"


def test_empty_index_returns_empty_results():
    store = make_store()

    assert store.search(np.array([1, 0, 0], dtype=np.float32)) == []


def test_invalid_dimension_is_rejected():
    store = make_store(dimension=3)

    with pytest.raises(ValueError, match="actual dimension"):
        store.add_vectors(np.ones((1, 2), dtype=np.float32), chunks(1))


def test_top_k_larger_than_index_size_returns_all_vectors():
    store = make_store()
    store.add_vectors(np.eye(3, dtype=np.float32), chunks(3))

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=20)

    assert len(results) == 3


def test_delete_removes_vector_and_metadata():
    store = make_store()
    store.add_vectors(np.eye(3, dtype=np.float32), chunks(3))

    assert store.delete([1]) == 1
    assert store.size == 2
    assert 1 not in [
        result.vector_id
        for result in store.search(np.array([0, 1, 0], dtype=np.float32), top_k=3)
    ]