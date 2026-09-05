"""FastAPI HTTP adapter for the injected legal RAG service."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
import os
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from source.legal_citation import Citation
from source.llm_answer_service import LegalAnswer
from source.rag_orchestrator import LegalRAGService, RAGStageError

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=10000, description="Vietnamese legal question")


class SourceDocumentResponse(BaseModel):
    document_id: str | None = None
    title: str | None = None
    document_number: str | None = None
    validity_status: str | None = None


class SourceChunkResponse(BaseModel):
    chunk_id: str | None = None
    text: str


class CitationResponse(BaseModel):
    citation_id: str
    label: str
    source_document: SourceDocumentResponse
    source_chunk: SourceChunkResponse
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    validity_status: str | None = None
    is_valid: bool
    is_trusted: bool
    used_in_answer: bool
    invalid_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    sources: list[dict[str, Any]]
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    stage: str | None = None


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or request.headers.get(
        "X-Request-ID",
        uuid4().hex,
    )


def _citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        citation_id=citation.citation_id,
        label=citation.label,
        source_document=SourceDocumentResponse(
            document_id=citation.source_document.document_id,
            title=citation.source_document.title,
            document_number=citation.source_document.document_number,
            validity_status=citation.source_document.validity_status,
        ),
        source_chunk=SourceChunkResponse(
            chunk_id=citation.source_chunk.chunk_id,
            text=citation.source_chunk.text,
        ),
        article=citation.article,
        clause=citation.clause,
        point=citation.point,
        validity_status=citation.validity_status,
        is_valid=citation.is_valid,
        is_trusted=citation.is_trusted,
        used_in_answer=citation.used_in_answer,
        invalid_reason=citation.invalid_reason,
    )


def _answer_response(answer: LegalAnswer, request_id: str, latency_ms: float) -> ChatResponse:
    metadata = {
        **dict(answer.retrieval_information),
        "request_id": request_id,
        "latency_ms": round(latency_ms, 3),
        "confidence": answer.confidence,
        "warnings": list(answer.warnings),
    }
    sources = [dict(source) for source in answer.referenced_documents]
    citations = [
        _citation_response(citation)
        for citation in answer.citations
        if isinstance(citation, Citation)
    ]
    return ChatResponse(
        answer=answer.answer,
        citations=citations,
        sources=sources,
        metadata=metadata,
    )


def create_app(
    rag_service: LegalRAGService | None = None,
    *,
    lifespan: Any | None = None,
) -> FastAPI:
    """Create the API app; production wiring can inject a configured service."""

    app = FastAPI(
        title="Vietnamese Legal Chatbot API",
        version="1.0.0",
        description="Grounded legal question answering over the RAG pipeline.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.rag_service = rag_service
    app.state.health_checker = None
    allowed_hosts = [
        host.strip()
        for host in os.getenv("ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    origins = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://127.0.0.1:5500,http://localhost:5500",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", uuid4().hex)
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-Ms"] = f"{(perf_counter() - started) * 1000:.3f}"
        return response

    async def service_dependency(request: Request) -> LegalRAGService:
        service = request.app.state.rag_service
        if service is None:
            raise RuntimeError("RAG service is not configured")
        return service

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        request_id = _request_id(request)
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request payload",
                "request_id": request_id,
                "details": exc.errors(),
            },
        )

    @app.exception_handler(RAGStageError)
    async def rag_error_handler(request: Request, exc: RAGStageError):
        request_id = _request_id(request)
        logger.exception("RAG request failed: request_id=%s stage=%s", request_id, exc.stage)
        return JSONResponse(
            status_code=502,
            content={
                "error": "rag_stage_error",
                "message": "The legal answer service could not complete the request.",
                "request_id": request_id,
                "stage": exc.stage,
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        request_id = _request_id(request)
        logger.exception("API runtime error: request_id=%s", request_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        request_id = _request_id(request)
        logger.exception("Unhandled API error: request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "Internal server error",
                "request_id": request_id,
            },
        )

    @app.get("/health", tags=["system"], summary="Health check")
    async def health(request: Request):
        checker = request.app.state.health_checker
        result = checker() if checker else {"status": "ok"}
        return {**result, "request_id": _request_id(request)}

    @app.get("/health/ready", tags=["system"], summary="Readiness check")
    async def readiness(request: Request):
        checker = request.app.state.health_checker
        result = checker() if checker else {"status": "ok"}
        status_code = 200 if result.get("status") == "ok" else 503
        return JSONResponse({**result, "request_id": _request_id(request)}, status_code=status_code)

    @app.get("/api/v1", tags=["system"], summary="API information")
    async def api_info(request: Request):
        return {"name": "Vietnamese Legal Chatbot API", "version": "v1", "request_id": _request_id(request)}

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        tags=["chat"],
        summary="Answer a legal question",
        description="Run the injected legal RAG service and return grounded answer metadata.",
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    async def chat(
        payload: ChatRequest,
        request: Request,
        service: LegalRAGService = Depends(service_dependency),
    ) -> ChatResponse:
        started = perf_counter()
        logger.info("Chat request started: request_id=%s", _request_id(request))
        answer = service.answer(payload.query)
        response = _answer_response(answer, _request_id(request), (perf_counter() - started) * 1000)
        logger.info(
            "Chat request completed: request_id=%s latency_ms=%.3f",
            _request_id(request),
            response.metadata["latency_ms"],
        )
        return response

    return app


app = create_app()