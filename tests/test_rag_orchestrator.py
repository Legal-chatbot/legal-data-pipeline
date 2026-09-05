import pytest

from source.llm_answer_service import LegalAnswer
from source.rag_orchestrator import LegalRAGService, RAGStageError


class Structured:
    intent = "article_lookup"


class QueryUnderstanding:
    def __init__(self, calls):
        self.calls = calls

    def understand(self, query):
        self.calls.append(("understand", query))
        return Structured()


class Retrieval:
    def __init__(self, calls):
        self.calls = calls

    def retrieve(self, query, *, structured_query):
        self.calls.append(("retrieve", query, structured_query))
        return type("RetrievalResult", (), {"chunks": ("chunk",)})()


class ContextBuilder:
    def __init__(self, calls):
        self.calls = calls

    def build(self, retrieval_result):
        self.calls.append(("context", retrieval_result))
        return type("Context", (), {"items": ("item",), "text": "legal context"})()


class AnswerGeneration:
    def __init__(self, calls):
        self.calls = calls

    def generate(self, query, structured_query, context):
        self.calls.append(("generate", query, structured_query, context))
        return LegalAnswer(answer="Câu trả lời", confidence=0.8)


def make_service(calls):
    return LegalRAGService(
        QueryUnderstanding(calls),
        Retrieval(calls),
        ContextBuilder(calls),
        AnswerGeneration(calls),
    )


def test_answer_runs_full_pipeline_in_order_and_records_trace():
    calls = []
    service = make_service(calls)

    result = service.answer("Điều 5 quy định gì?")

    assert result.answer == "Câu trả lời"
    assert [call[0] for call in calls] == ["understand", "retrieve", "context", "generate"]
    trace = result.retrieval_information["rag_trace"]
    assert [stage["stage"] for stage in trace["stages"]] == [
        "query_understanding",
        "retrieval",
        "context_builder",
        "answer_generation",
    ]
    assert all(stage["status"] == "ok" for stage in trace["stages"])
    assert all(stage["duration_ms"] >= 0 for stage in trace["stages"])
    assert service.last_trace.request_id == trace["request_id"]


def test_stage_failure_is_wrapped_and_trace_is_preserved():
    class BrokenRetrieval(Retrieval):
        def retrieve(self, query, *, structured_query):
            raise TimeoutError("retrieval timeout")

    calls = []
    service = LegalRAGService(
        QueryUnderstanding(calls),
        BrokenRetrieval(calls),
        ContextBuilder(calls),
        AnswerGeneration(calls),
    )

    with pytest.raises(RAGStageError, match="retrieval") as raised:
        service.answer("câu hỏi")

    assert raised.value.stage == "retrieval"
    assert service.last_trace.stages[-1].status == "error"
    assert [stage.stage for stage in service.last_trace.stages] == [
        "query_understanding",
        "retrieval",
    ]


def test_empty_query_is_rejected_before_pipeline():
    service = make_service([])

    with pytest.raises(ValueError):
        service.answer(" ")