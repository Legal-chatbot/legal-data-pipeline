from fastapi.testclient import TestClient

from source.fastapi_rag_api import create_app
from source.legal_citation import Citation, SourceChunk, SourceDocument
from source.llm_answer_service import LegalAnswer


def citation():
    document = SourceDocument("d1", "Luật thử nghiệm", "01/2026/QH15", "Còn hiệu lực")
    return Citation(
        citation_id="C1",
        label="Luật thử nghiệm - Điều 5",
        source_document=document,
        source_chunk=SourceChunk("c1", "Điều 5. Nội dung.", document),
        article="5",
        validity_status="Còn hiệu lực",
        is_trusted=True,
        used_in_answer=True,
    )


class FakeRAG:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return LegalAnswer(
            answer="Căn cứ Điều 5 [C1].",
            citations=(citation(),),
            referenced_documents=({"id": "d1", "title": "Luật thử nghiệm"},),
            confidence=0.9,
            retrieval_information={"context_count": 1},
        )


class BrokenRAG:
    def answer(self, query):
        raise RuntimeError("RAG service is not configured")


def test_chat_calls_only_injected_service_and_returns_contract():
    service = FakeRAG()
    client = TestClient(create_app(service))

    response = client.post(
        "/api/v1/chat",
        json={"query": "Điều 5 quy định gì?"},
        headers={"X-Request-ID": "request-123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Căn cứ Điều 5 [C1]."
    assert body["citations"][0]["citation_id"] == "C1"
    assert body["sources"][0]["title"] == "Luật thử nghiệm"
    assert body["metadata"]["request_id"] == "request-123"
    assert body["metadata"]["latency_ms"] >= 0
    assert response.headers["X-Request-ID"] == "request-123"
    assert service.queries == ["Điều 5 quy định gì?"]


def test_health_and_api_info():
    client = TestClient(create_app(FakeRAG()))

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/v1").json()["version"] == "v1"
    assert client.get("/openapi.json").status_code == 200


def test_validation_error_and_service_error():
    client = TestClient(create_app(FakeRAG()))
    invalid = client.post("/api/v1/chat", json={"query": ""})
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"

    unavailable = TestClient(create_app(BrokenRAG())).post(
        "/api/v1/chat", json={"query": "câu hỏi"}
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["error"] == "service_unavailable"