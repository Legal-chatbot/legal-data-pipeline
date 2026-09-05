import numpy as np

from source.embedding_service import LegalChunk
from source.faiss_vector_store import SearchResult
from source.hybrid_retrieval import (
    GraphSearchResult,
    HybridRetrievalConfig,
    HybridRetrievalEngine,
    RetrievalWeights,
)


class FakeUnderstanding:
    def understand(self, query):
        return type("Structured", (), {
            "intent": "article_lookup",
            "article_references": (),
            "document_identifiers": (),
            "normalized_entities": {},
        })()


class FakeEmbedding:
    def embed_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_embedding, top_k):
        return self.results[:top_k]


class FakeGraph:
    def __init__(self, results):
        self.results = results

    def search(self, structured_query, top_k):
        return self.results[:top_k]


def vector_result(chunk, score):
    return SearchResult(vector_id=score, score=score, chunk=chunk)


def graph_result(chunk, score, entity_match_score=0.0):
    return GraphSearchResult(
        chunk=chunk,
        score=score,
        document={"id": chunk.metadata.get("document_id", "doc-1")},
        entity_match_score=entity_match_score,
    )


def engine(vector=None, graph=None, weights=None):
    return HybridRetrievalEngine(
        embedding_service=FakeEmbedding() if vector is not None else None,
        vector_store=FakeVectorStore(vector or []) if vector is not None else None,
        graph_retriever=FakeGraph(graph or []) if graph is not None else None,
        query_understanding=FakeUnderstanding() if graph is not None else None,
        config=HybridRetrievalConfig(weights=weights or RetrievalWeights()),
    )


def test_vector_only():
    chunk = LegalChunk("vector chunk", chunk_id="v1")

    result = engine(vector=[vector_result(chunk, 0.9)]).retrieve("câu hỏi")

    assert result.chunks == (chunk,)
    assert result.retrieval_sources == (("vector",),)


def test_graph_only():
    chunk = LegalChunk("graph chunk", chunk_id="g1")

    result = engine(graph=[graph_result(chunk, 0.8, 1.0)]).retrieve("Điều 5")

    assert result.chunks == (chunk,)
    assert result.retrieval_sources == (("graph",),)
    assert result.explanations[0]["graph_raw_score"] == 0.8


def test_both_sources_are_fused_and_duplicate_is_not_returned_twice():
    shared = LegalChunk("same chunk", chunk_id="same")
    vector_only = LegalChunk("vector only", chunk_id="v")
    graph_only = LegalChunk("graph only", chunk_id="g")
    result = engine(
        vector=[vector_result(shared, 0.9), vector_result(vector_only, 0.4)],
        graph=[graph_result(shared, 0.8, 1.0), graph_result(graph_only, 0.5)],
    ).retrieve("Điều 5", top_k=5)

    assert len(result.chunks) == 3
    assert result.chunks[0].chunk_id == "same"
    assert result.retrieval_sources[0] == ("graph", "vector")


def test_empty_retrieval():
    result = engine(vector=[], graph=[]).retrieve("Điều 5")

    assert result.chunks == ()
    assert result.scores == ()


def test_configurable_weights_resolve_conflicting_ranking():
    vector_winner = LegalChunk("vector winner", chunk_id="v")
    graph_winner = LegalChunk("graph winner", chunk_id="g")
    result = engine(
        vector=[vector_result(vector_winner, 1.0), vector_result(graph_winner, 0.0)],
        graph=[graph_result(vector_winner, 0.0), graph_result(graph_winner, 1.0)],
        weights=RetrievalWeights(vector=0.1, graph=0.9, entity_match=0.0),
    ).retrieve("Điều 5", top_k=2)

    assert result.chunks[0].chunk_id == "g"