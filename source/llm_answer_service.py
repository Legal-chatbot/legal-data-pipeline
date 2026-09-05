"""LLM answer generation behind a provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib import error, request


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot return a valid completion."""


class LLMProvider(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        timeout_seconds: float,
    ) -> str:
        ...


@dataclass(frozen=True)
class LLMProviderConfig:
    api_key: str | None = None
    api_url: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    api_key_env: str = "LLM_API_KEY"

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY"),
            api_url=os.getenv("LLM_API_URL", cls.api_url),
            model=os.getenv("LLM_MODEL", cls.model),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")


class OpenAICompatibleProvider:
    """Small standard-library client for OpenAI-compatible chat APIs."""

    def __init__(self, config: LLMProviderConfig | None = None) -> None:
        self.config = config or LLMProviderConfig.from_env()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise LLMProviderError(
                f"missing API key; configure {self.config.api_key_env}"
            )
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = timeout_seconds or self.config.timeout_seconds
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                http_request = request.Request(
                    self.config.api_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with request.urlopen(http_request, timeout=timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = response_payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise LLMProviderError("LLM response did not contain text content")
                return content.strip()
            except error.HTTPError as exc:
                last_error = exc
                if not self._is_retryable_status(exc.code) or attempt == attempts - 1:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise LLMProviderError(
                        f"LLM API request failed with HTTP {exc.code}: {detail}"
                    ) from exc
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise LLMProviderError("LLM API request failed") from exc
            if self.config.retry_backoff_seconds:
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise LLMProviderError("LLM API request failed") from last_error

    @staticmethod
    def _is_retryable_status(status: int) -> bool:
        return status == 408 or status == 429 or 500 <= status < 600


SYSTEM_PROMPT = """You are a careful legal information assistant.

Rules:
1. Use only the legal context provided in the user message.
2. Never invent, complete, or assume an article, legal rule, date, or document.
3. If the context is insufficient, say clearly that the provided context is insufficient.
4. Cite the provided source metadata for every material legal statement.
5. Distinguish directly supported legal information from reasoning or interpretation.
6. Do not make an unsupported legal assertion. When uncertain, state the limitation.

Answer in Vietnamese unless the user asks for another language. Keep citations attached
to the relevant statements and do not cite sources that are absent from the context.
"""


@dataclass(frozen=True)
class LegalAnswer:
    answer: str
    citations: tuple[str, ...] = field(default_factory=tuple)
    referenced_documents: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    confidence: float | None = None
    retrieval_information: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


class LLMAnswerGenerationService:
    """Create a grounded legal answer without coupling the app to one LLM API."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        user_query: str,
        structured_query: Any,
        retrieved_context: Any,
    ) -> LegalAnswer:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        context_text = self._context_text(retrieved_context)
        citations = self._context_citations(retrieved_context)
        documents = self._context_documents(retrieved_context)
        retrieval_information = self._retrieval_information(retrieved_context)
        if not context_text.strip():
            return LegalAnswer(
                answer="Không đủ ngữ cảnh pháp lý được cung cấp để trả lời câu hỏi này.",
                citations=tuple(citations),
                referenced_documents=tuple(documents),
                confidence=0.0,
                retrieval_information=retrieval_information,
                warnings=("Retrieved legal context is empty.",),
            )

        structured = self._serialize_structured_query(structured_query)
        user_prompt = (
            f"USER QUERY:\n{user_query.strip()}\n\n"
            f"STRUCTURED QUERY:\n{structured}\n\n"
            f"LEGAL CONTEXT:\n{context_text}\n\n"
            "Return a grounded answer with inline citations from the context. "
            "Explicitly label reasoning when interpretation is needed."
        )
        try:
            answer = self.provider.complete(
                SYSTEM_PROMPT,
                user_prompt,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            if isinstance(exc, LLMProviderError):
                raise
            raise LLMProviderError("legal answer generation failed") from exc
        warnings = ()
        if not citations:
            warnings = ("Context has no citation metadata; verify the answer manually.",)
        return LegalAnswer(
            answer=answer,
            citations=tuple(citations),
            referenced_documents=tuple(documents),
            confidence=self._confidence(retrieved_context),
            retrieval_information=retrieval_information,
            warnings=warnings,
        )

    @staticmethod
    def _context_text(context: Any) -> str:
        if isinstance(context, str):
            return context
        return str(getattr(context, "text", ""))

    @staticmethod
    def _context_items(context: Any) -> Sequence[Any]:
        return tuple(getattr(context, "items", ()))

    @classmethod
    def _context_citations(cls, context: Any) -> list[str]:
        return list(
            dict.fromkeys(
                str(item.citation)
                for item in cls._context_items(context)
                if getattr(item, "citation", None)
            )
        )

    @classmethod
    def _context_documents(cls, context: Any) -> list[Mapping[str, Any]]:
        documents = []
        for item in cls._context_items(context):
            document = {
                key: value
                for key, value in {
                    "document_id": getattr(item, "document_id", None),
                    "title": getattr(item, "document_title", None),
                    "so_ky_hieu": getattr(item, "document_number", None),
                    "validity_status": getattr(item, "validity_status", None),
                }.items()
                if value is not None
            }
            if document and document not in documents:
                documents.append(document)
        return documents

    @classmethod
    def _retrieval_information(cls, context: Any) -> dict[str, Any]:
        items = cls._context_items(context)
        return {
            "context_count": len(items),
            "scores": [getattr(item, "retrieval_score", None) for item in items],
            "sources": [list(getattr(item, "retrieval_source", ())) for item in items],
        }

    @staticmethod
    def _confidence(context: Any) -> float | None:
        scores = [
            float(item.retrieval_score)
            for item in LLMAnswerGenerationService._context_items(context)
            if getattr(item, "retrieval_score", None) is not None
        ]
        return max(0.0, min(1.0, max(scores))) if scores else None

    @staticmethod
    def _serialize_structured_query(query: Any) -> str:
        if hasattr(query, "to_dict"):
            query = query.to_dict()
        if query is None:
            return "{}"
        return json.dumps(query, ensure_ascii=False, default=str, indent=2)