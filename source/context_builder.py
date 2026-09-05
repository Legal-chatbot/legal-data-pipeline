"""Build bounded legal context for the answer generation stage."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class ContextBuilderConfig:
    max_contexts: int = 5
    max_characters: int = 12000
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_contexts < 1 or self.max_characters < 1:
            raise ValueError("context limits must be greater than zero")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")


@dataclass(frozen=True)
class ContextItem:
    document_id: str | None
    document_title: str | None
    document_number: str | None
    article: str | None
    clause: str | None
    point: str | None
    validity_status: str | None
    retrieval_score: float
    retrieval_source: tuple[str, ...]
    text: str
    priority: float
    citation: str
    chunk_id: str | None = None


@dataclass(frozen=True)
class LLMContext:
    text: str
    items: tuple[ContextItem, ...]
    used_characters: int
    estimated_tokens: int


class ContextBuilder:
    """Format retrieval results while retaining source and legal structure metadata."""

    _reference_pattern = re.compile(
        r"(?:(?:điểm)\s+([a-zđ])\s+)?"
        r"(?:(?:khoản)\s+(\d+[a-z]?|[ivx]+)\s+)?"
        r"(?:điều)\s+(\d+[a-z]?)",
        re.IGNORECASE,
    )

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self.config = config or ContextBuilderConfig()

    def build(self, retrieval_result: Any) -> LLMContext:
        items = []
        seen = set()
        chunks = tuple(getattr(retrieval_result, "chunks", ()))
        documents = tuple(getattr(retrieval_result, "documents", ()))
        scores = tuple(getattr(retrieval_result, "scores", ()))
        sources = tuple(getattr(retrieval_result, "retrieval_sources", ()))
        for index, chunk in enumerate(chunks):
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            document = dict(documents[index]) if index < len(documents) else metadata
            chunk_id = getattr(chunk, "chunk_id", None) or metadata.get("chunk_id")
            key = chunk_id or chunk.text.strip()
            if key in seen:
                continue
            seen.add(key)
            article, clause, point = self._structure(chunk.text, metadata)
            document_id = self._first(document, metadata, "document_id", "id", "doc_id")
            title = self._first(document, metadata, "title", "document_title")
            number = self._first(document, metadata, "so_ky_hieu", "document_number")
            validity = self._first(document, metadata, "tinh_trang_hieu_luc", "validity_status")
            score = float(scores[index]) if index < len(scores) else 0.0
            source = tuple(sources[index]) if index < len(sources) else ()
            citation = self._citation(title, number, article, clause, point)
            items.append(ContextItem(
                document_id=str(document_id) if document_id is not None else None,
                document_title=str(title) if title is not None else None,
                document_number=str(number) if number is not None else None,
                article=article, clause=clause, point=point,
                validity_status=str(validity) if validity is not None else None,
                retrieval_score=score, retrieval_source=source,
                text=chunk.text.strip(), priority=score, citation=citation,
                chunk_id=str(chunk_id) if chunk_id is not None else None,
            ))
        items.sort(key=lambda item: item.priority, reverse=True)
        selected = []
        blocks = []
        for item in items[: self.config.max_contexts]:
            block = self._render(item, len(selected) + 1)
            if len("\n\n".join(blocks + [block])) > self.config.max_characters:
                remaining = self.config.max_characters - len("\n\n".join(blocks)) - 2
                text_budget = max(0, remaining - len(block) + len(item.text))
                text = self._truncate(item.text, text_budget)
                if not text:
                    continue
                item = ContextItem(**{**item.__dict__, "text": text})
                block = self._render(item, len(selected) + 1)
            if self.config.max_tokens and len("\n\n".join(blocks + [block])) // 4 > self.config.max_tokens:
                continue
            selected.append(item)
            blocks.append(block)
        text = "\n\n".join(blocks)
        return LLMContext(text, tuple(selected), len(text), (len(text) + 3) // 4)

    @staticmethod
    def _first(primary: Mapping[str, Any], secondary: Mapping[str, Any], *keys: str) -> Any:
        for mapping in (primary, secondary):
            for key in keys:
                if mapping.get(key) is not None:
                    return mapping[key]
        return None

    def _structure(self, text: str, metadata: Mapping[str, Any]):
        articles = metadata.get("articles")
        if isinstance(articles, str):
            articles = [value.strip() for value in articles.split(",") if value.strip()]
        match = self._reference_pattern.search(text)
        if match:
            point, clause, article = match.groups()
            return article.lower(), clause.lower() if clause else None, point.casefold() if point else None
        return (str(articles[0]) if articles else None), None, None

    @staticmethod
    def _citation(title, number, article, clause, point):
        name = title or number or "Unknown document"
        location = " ".join(value for value in (
            f"Điều {article}" if article else None,
            f"khoản {clause}" if clause else None,
            f"điểm {point}" if point else None,
        ) if value)
        return f"{name}{' - ' + location if location else ''}"

    @staticmethod
    def _render(item: ContextItem, number: int) -> str:
        return "\n".join((
            f"[CONTEXT {number}]",
            f"DOCUMENT: {item.document_title or item.document_id or 'Unknown document'}",
            f"SO_KY_HIEU: {item.document_number or 'Unknown'}",
            f"ARTICLE: {item.article or 'Unknown'}",
            f"CLAUSE: {item.clause or 'Unknown'}",
            f"POINT: {item.point or 'Unknown'}",
            f"VALIDITY: {item.validity_status or 'Unknown'}",
            f"RETRIEVAL_SCORE: {item.retrieval_score:.6f}",
            f"RETRIEVAL_SOURCE: {', '.join(item.retrieval_source) or 'unknown'}",
            f"CITATION: {item.citation}",
            "TEXT:", item.text, "[/CONTEXT]",
        ))

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        boundary = max(text.rfind("\n\n", 0, limit), text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        return text[:boundary].rstrip() + " [...]" if boundary > 0 else ""