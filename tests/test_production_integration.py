from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from source.fastapi_rag_api import create_app
from source.llm_answer_service import LegalAnswer


class FakeService:
    def answer(self, query):
        return LegalAnswer(answer=f"Đã xử lý: {query}")


def test_startup_health_chat_shutdown_lifecycle():
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        app.state.rag_service = FakeService()
        app.state.health_checker = lambda: {"status": "ok", "neo4j_connected": True, "faiss_loaded": True}
        yield
        events.append("shutdown")

    with TestClient(create_app(lifespan=lifespan)) as client:
        health = client.get("/health/ready")
        response = client.post("/api/v1/chat", json={"query": "Điều 5?"})

    assert events == ["startup", "shutdown"]
    assert health.status_code == 200
    assert health.json()["neo4j_connected"] is True
    assert response.status_code == 200
    assert response.json()["answer"] == "Đã xử lý: Điều 5?"