"""Hybrid retrieval over FAISS vectors and Neo4j legal graph results."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Protocol, Sequence

from source.embedding_service import EmbeddingService, LegalChunk
from source.faiss_vector_store import FaissVectorStore, SearchResult
from source.query_understanding import QueryUnderstandingService, StructuredQuery

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphSearchResult:
    chunk: LegalChunk
    score: float
    document: Mapping[str, Any] = field(default_factory=dict)
    entity_match_score: float = 0.0
    relationship_types: tuple[str, ...] = field(default_factory=tuple)
    explanation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalWeights:
    vector: float = 0.6
    graph: float = 0.4
    entity_match: float = 0.2

    def __post_init__(self) -> None:
        if self.vector < 0 or self.graph < 0 or self.entity_match < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.vector == 0 and self.graph == 0:
            raise ValueError("vector or graph weight must be greater than zero")


@dataclass(frozen=True)
class HybridRetrievalConfig:
    top_k: int = 5
    vector_top_k: int = 20
    graph_top_k: int = 20
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.vector_top_k < 1 or self.graph_top_k < 1:
            raise ValueError("retrieval top_k values must be greater than zero")


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[LegalChunk, ...]
    documents: tuple[Mapping[str, Any], ...]
    scores: tuple[float, ...]
    retrieval_sources: tuple[tuple[str, ...], ...]
    explanations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    structured_query: StructuredQuery | None = None


class GraphRetriever(Protocol):
    def search(self, structured_query: StructuredQuery, top_k: int) -> Sequence[GraphSearchResult]:
        ...


class Neo4jGraphRetriever:
    """Retrieve graph candidates using intent, entities, articles and relations."""

    QUERY = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    OPTIONAL MATCH (d)-[relation]->(related:Document)
    WITH c, d, collect(DISTINCT type(relation)) AS relationship_types
    WHERE (
        any(article IN coalesce(c.articles, []) WHERE article IN $articles)
        OR d.id IN $document_ids
        OR d.so_ky_hieu IN $document_numbers
        OR any(term IN $law_names WHERE toLower(coalesce(d.title, '')) CONTAINS term)
        OR ($intent = 'validity_lookup' AND d.tinh_trang_hieu_luc IS NOT NULL)
        OR ($intent = 'amendment_lookup' AND size(relationship_types) > 0)
    )
    RETURN c { .* } AS chunk,
           d { .* } AS document,
           relationship_types
    LIMIT $top_k
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def search(self, structured_query: StructuredQuery, top_k: int) -> list[GraphSearchResult]:
        article_numbers = [reference.article for reference in structured_query.article_references]
        document_numbers = list(structured_query.document_identifiers)
        document_ids = list(
            structured_query.normalized_entities.get("document_id", ())
        )
        law_names = list(structured_query.normalized_entities.get("law_name", ()))
        params = {
            "articles": article_numbers,
            "document_ids": document_ids,
            "document_numbers": document_numbers,
            "law_names": law_names,
            "intent": structured_query.intent,
            "top_k": top_k,
        }

        def read(tx, values):
            return list(tx.run(self.QUERY, **values))

        rows = self.connection.execute_read(read, params)
        results = []
        for row in rows:
            chunk = row["chunk"]
            document = row["document"]
            relationship_types = tuple(row.get("relationship_types", ()))
            article_match = any(
                article in article_numbers
                for article in chunk.get("articles", ())
            )
            document_match = (
                document.get("id") in document_ids
                or document.get("so_ky_hieu") in document_numbers
            )
            entity_match_score = 1.0 if article_match or document_match else 0.0
            base_score = 1.0 if entity_match_score else 0.5
            if structured_query.intent == "validity_lookup" and document.get("tinh_trang_hieu_luc"):
                base_score += 0.25
            if relationship_types:
                base_score += 0.25
            results.append(
                GraphSearchResult(
                    chunk=LegalChunk(
                        text=chunk.get("text", ""),
                        chunk_id=chunk.get("id"),
                        metadata={"document_id": document.get("id"), **chunk},
                    ),
                    score=base_score,
                    document=document,
                    entity_match_score=entity_match_score,
                    relationship_types=relationship_types,
                    explanation={
                        "article_match": article_match,
                        "document_match": document_match,
                        "validity_present": bool(document.get("tinh_trang_hieu_luc")),
                    },
                )
            )
        return results


class HybridRetrievalEngine:
    """Understand a query, retrieve from both backends, then rank fused chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
        graph_retriever: GraphRetriever | None = None,
        config: HybridRetrievalConfig | None = None,
        query_understanding: Any | None = None,
    ) -> None:
        self.embedding_service = embedding_service or (
            EmbeddingService() if vector_store is not None else None
        )
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.config = config or HybridRetrievalConfig()
        self.query_understanding = query_understanding or QueryUnderstandingService()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        structured_query: StructuredQuery | None = None,
    ) -> RetrievalResult:
        if structured_query is None and self.query_understanding is not None:
            structured_query = self.query_understanding.understand(query)

        vector_results: Sequence[SearchResult] = ()
        if self.vector_store is not None:
            if self.embedding_service is None:
                raise ValueError("embedding_service is required for vector retrieval")
            query_embedding = self.embedding_service.embed_query(query)
            vector_results = self.vector_store.search(
                query_embedding,
                top_k=self.config.vector_top_k,
            )

        graph_results: Sequence[GraphSearchResult] = ()
        if self.graph_retriever is not None:
            if structured_query is None:
                raise ValueError("query_understanding is required for graph retrieval")
            graph_results = self.graph_retriever.search(
                structured_query,
                top_k=self.config.graph_top_k,
            )

        return self._fuse(vector_results, graph_results, structured_query, top_k)

    def _fuse(
        self,
        vector_results: Sequence[SearchResult],
        graph_results: Sequence[GraphSearchResult],
        structured_query: StructuredQuery | None,
        top_k: int | None,
    ) -> RetrievalResult:
        candidates: dict[str, dict[str, Any]] = {}
        vector_scores = self._normalize([result.score for result in vector_results])
        graph_scores = self._normalize([result.score for result in graph_results])
        for result, normalized_score in zip(vector_results, vector_scores):
            key = self._chunk_key(result.chunk)
            candidate = candidates.setdefault(key, self._new_candidate(result.chunk))
            candidate["vector_score"] = normalized_score
            candidate["sources"].add("vector")
            candidate["explanation"]["vector_raw_score"] = result.score
        for result, normalized_score in zip(graph_results, graph_scores):
            key = self._chunk_key(result.chunk)
            candidate = candidates.setdefault(key, self._new_candidate(result.chunk))
            candidate["graph_score"] = normalized_score
            candidate["entity_match"] = result.entity_match_score
            candidate["sources"].add("graph")
            candidate["document"] = dict(result.document)
            candidate["explanation"].update(result.explanation)
            candidate["explanation"]["graph_raw_score"] = result.score
            candidate["explanation"]["relationship_types"] = result.relationship_types

        ranked = sorted(candidates.values(), key=self._fused_score, reverse=True)
        limit = self.config.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be greater than zero")
        ranked = ranked[:limit]
        return RetrievalResult(
            chunks=tuple(candidate["chunk"] for candidate in ranked),
            documents=tuple(candidate["document"] for candidate in ranked),
            scores=tuple(self._fused_score(candidate) for candidate in ranked),
            retrieval_sources=tuple(
                tuple(sorted(candidate["sources"])) for candidate in ranked
            ),
            explanations=tuple(candidate["explanation"] for candidate in ranked),
            structured_query=structured_query,
        )

    def _fused_score(self, candidate: Mapping[str, Any]) -> float:
        weights = self.config.weights
        available = []
        if "vector" in candidate["sources"]:
            available.append((weights.vector, candidate["vector_score"]))
        if "graph" in candidate["sources"]:
            available.append((weights.graph, candidate["graph_score"]))
        weight_total = sum(weight for weight, _ in available)
        score = sum(weight * value for weight, value in available) / weight_total
        if "graph" in candidate["sources"]:
            score += weights.entity_match * candidate["entity_match"]
        return score

    @staticmethod
    def _normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [1.0 for _ in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    @staticmethod
    def _chunk_key(chunk: LegalChunk) -> str:
        return f"id:{chunk.chunk_id}" if chunk.chunk_id else f"text:{chunk.text}"

    @staticmethod
    def _new_candidate(chunk: LegalChunk) -> dict[str, Any]:
        return {
            "chunk": chunk,
            "document": dict(chunk.metadata),
            "sources": set(),
            "vector_score": 0.0,
            "graph_score": 0.0,
            "entity_match": 0.0,
            "explanation": {},
        }