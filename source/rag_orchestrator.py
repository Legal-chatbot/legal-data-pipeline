"""Application orchestration for the legal RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from time import perf_counter
from typing import Any, Mapping, Protocol
from uuid import uuid4

from source.llm_answer_service import LegalAnswer

logger = logging.getLogger(__name__)


class RAGStageError(RuntimeError):
    """Raised when a named RAG pipeline stage fails."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"RAG stage failed: {stage}")
        self.stage = stage
        self.cause = cause


class QueryUnderstandingComponent(Protocol):
    def understand(self, query: str) -> Any:
        ...


class RetrievalComponent(Protocol):
    def retrieve(self, query: str, *, structured_query: Any = None) -> Any:
        ...


class ContextBuilderComponent(Protocol):
    def build(self, retrieval_result: Any) -> Any:
        ...


class AnswerGenerationComponent(Protocol):
    def generate(self, user_query: str, structured_query: Any, retrieved_context: Any) -> LegalAnswer:
        ...


@dataclass(frozen=True)
class RAGStageTrace:
    stage: str
    duration_ms: float
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGTrace:
    request_id: str
    stages: tuple[RAGStageTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stages": [
                {
                    "stage": stage.stage,
                    "duration_ms": stage.duration_ms,
                    "status": stage.status,
                    "details": dict(stage.details),
                }
                for stage in self.stages
            ],
        }


class LegalRAGService:
    """Coordinate pipeline components without owning their retrieval logic."""

    def __init__(
        self,
        query_understanding: QueryUnderstandingComponent,
        retrieval: RetrievalComponent,
        context_builder: ContextBuilderComponent,
        answer_generation: AnswerGenerationComponent,
        *,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        self.query_understanding = query_understanding
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.answer_generation = answer_generation
        self.logger = logger_instance or logger
        self.last_trace: RAGTrace | None = None

    def answer(self, query: str) -> LegalAnswer:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        request_id = uuid4().hex
        traces: list[RAGStageTrace] = []
        try:
            structured_query = self._run_stage(
                "query_understanding",
                traces,
                lambda: self.query_understanding.understand(query),
            )
            retrieval_result = self._run_stage(
                "retrieval",
                traces,
                lambda: self.retrieval.retrieve(
                    query,
                    structured_query=structured_query,
                ),
            )
            context = self._run_stage(
                "context_builder",
                traces,
                lambda: self.context_builder.build(retrieval_result),
            )
            legal_answer = self._run_stage(
                "answer_generation",
                traces,
                lambda: self.answer_generation.generate(query, structured_query, context),
            )
        except RAGStageError:
            self.last_trace = RAGTrace(request_id, tuple(traces))
            raise

        trace = RAGTrace(request_id, tuple(traces))
        self.last_trace = trace
        retrieval_information = {
            **dict(legal_answer.retrieval_information),
            "rag_trace": trace.to_dict(),
        }
        self.logger.info(
            "RAG request completed: request_id=%s stages=%d",
            request_id,
            len(traces),
        )
        return replace(legal_answer, retrieval_information=retrieval_information)

    def _run_stage(self, stage: str, traces: list[RAGStageTrace], operation):
        started = perf_counter()
        try:
            result = operation()
        except Exception as exc:
            duration_ms = (perf_counter() - started) * 1000
            traces.append(
                RAGStageTrace(
                    stage=stage,
                    duration_ms=duration_ms,
                    status="error",
                    details={"error_type": type(exc).__name__},
                )
            )
            self.logger.exception("RAG stage failed: %s", stage)
            raise RAGStageError(stage, exc) from exc
        duration_ms = (perf_counter() - started) * 1000
        details = self._stage_details(result)
        traces.append(
            RAGStageTrace(
                stage=stage,
                duration_ms=duration_ms,
                status="ok",
                details=details,
            )
        )
        self.logger.info("RAG stage completed: %s duration_ms=%.2f", stage, duration_ms)
        return result

    @staticmethod
    def _stage_details(result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        if hasattr(result, "chunks"):
            return {"chunk_count": len(result.chunks)}
        if hasattr(result, "items"):
            return {"context_count": len(result.items)}
        if hasattr(result, "intent"):
            return {"intent": result.intent}
        if isinstance(result, LegalAnswer):
            return {"warning_count": len(result.warnings)}
        return {}