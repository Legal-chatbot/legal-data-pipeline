"""Build structured, citation-preserving context for an LLM prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from source.hybrid_retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextBuilderConfig:
    max_contexts: int = 5
    max_characters: int = 12000
    max_tokens: int | None = None
    source_priority: Mapping[str, float] = field(default_factory=dict)
    article_match_priority: float = 0.0

    def __post_init__(self) -> None:
        if self.max_contexts < 1:
            raise ValueError("max_contexts must be greater than zero")
        if self.max_characters < 1:
            raise ValueError("max_characters must be greater than zero")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        if self.article_match_priority < 0:
            raise ValueError("article_match_priority cannot be negative")
        if any(priority < 0 for priority in self.source_priority.values()):
            raise ValueError("source priorities cannot be negative")


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
    truncated: bool = False

    def render(self, number: int) -> str:
        lines = [
            f"[CONTEXT {number}]",
            f"DOCUMENT: {self.document_title or self.document_id or 'Unknown document'}",
            f"SO_KY_HIEU: {self.document_number or 'Unknown'}",
            f"ARTICLE: {self.article or 'Unknown'}",
            f"CLAUSE: {self.clause or 'Unknown'}",
            f"POINT: {self.point or 'Unknown'}",
            f"VALIDITY: {self.validity_status or 'Unknown'}",
            f"RETRIEVAL_SCORE: {self.retrieval_score:.6f}",
            f"RETRIEVAL_SOURCE: {', '.join(self.retrieval_source) or 'unknown'}",
            f"CITATION: {self.citation}",
            "TEXT:",
            self.text,
            "[/CONTEXT]",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class LLMContext:
    text: str
    items: tuple[ContextItem, ...]
    used_characters: int
    estimated_tokens: int

    def to_prompt(self) -> str:
        return self.text


class ContextBuilder:
    """Turn hybrid retrieval output into bounded, structured LLM context."""

    _structure_pattern = re.compile(
        r"(?:(?:điểm)\s+([a-zđ])\s+)?"
        r"(?:(?:khoản)\s+(\d+[a-z]?|[ivx]+)\s+)?"
        r"(?:điều)\s+(\d+[a-z]?)",
        re.IGNORECASE,
    )

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self.config = config or ContextBuilderConfig()

    def build(self, retrieval: RetrievalResult) -> LLMContext:
        candidates = self._deduplicate(retrieval)
        candidates.sort(key=lambda item: item.priority, reverse=True)
        selected: list[ContextItem] = []
        rendered: list[str] = []
        used_characters = 0
        for candidate in candidates:
            if len(selected) >= self.config.max_contexts:
                break
            remaining = self.config.max_characters - used_characters
            if rendered:
                remaining -= 2
            item = self._fit_item(candidate, remaining, len(selected) + 1)
            if item is None:
                continue
            block = item.render(len(selected) + 1)
            if self.config.max_tokens is not None:
                projected_tokens = self._estimate_tokens("\n\n".join(rendered + [block]))
                if projected_tokens > self.config.max_tokens:
                    token_remaining = self.config.max_tokens - self._estimate_tokens("\n\n".join(rendered))
                    item = self._fit_to_tokens(candidate, remaining, token_remaining, len(selected) + 1)
                    if item is None:
                        continue
                    block = item.render(len(selected) + 1)
                    projected_tokens = self._estimate_tokens("\n\n".join(rendered + [block]))
                    if projected_tokens > self.config.max_tokens:
                        continue
            selected.append(item)
            rendered.append(block)
            used_characters += len(block) + (2 if len(rendered) > 1 else 0)
        text = "\n\n".join(rendered)
        return LLMContext(
            text=text,
            items=tuple(selected),
            used_characters=len(text),
            estimated_tokens=self._estimate_tokens(text),
        )

    def _deduplicate(self, retrieval: RetrievalResult) -> list[ContextItem]:
        candidates: dict[str, ContextItem] = {}
        for index, chunk in enumerate(retrieval.chunks):
            document = dict(retrieval.documents[index]) if index < len(retrieval.documents) else {}
            sources = retrieval.retrieval_sources[index] if index < len(retrieval.retrieval_sources) else ()
            score = retrieval.scores[index] if index < len(retrieval.scores) else 0.0
            article, clause, point = self._extract_structure(chunk, document)
            document_id = self._first(document, chunk.metadata, "id", "document_id", "doc_id")
            title = self._first(document, chunk.metadata, "title", "document_title")
            number = self._first(document, chunk.metadata, "so_ky_hieu", "document_number")
            validity = self._first(
                document,
                chunk.metadata,
                "tinh_trang_hieu_luc",
                "validity_status",
            )
            reference_articles = {
                reference.article
                for reference in (retrieval.structured_query.article_references if retrieval.structured_query else ())
            }
            article_match = article in reference_articles if article else False
            priority = score + sum(
                self.config.source_priority.get(source, 0.0) for source in sources
            )
            if article_match:
                priority += self.config.article_match_priority
            citation = self._citation(title, number, article, clause, point)
            item = ContextItem(
                document_id=str(document_id) if document_id is not None else None,
                document_title=str(title) if title is not None else None,
                document_number=str(number) if number is not None else None,
                article=article,
                clause=clause,
                point=point,
                validity_status=str(validity) if validity is not None else None,
                retrieval_score=float(score),
                retrieval_source=tuple(sorted(set(sources))),
                text=chunk.text.strip(),
                priority=priority,
                citation=citation,
            )
            key = f"id:{chunk.chunk_id}" if chunk.chunk_id else f"text:{chunk.text.strip()}"
            previous = candidates.get(key)
            if previous is None or item.priority > previous.priority:
                candidates[key] = self._merge_items(item, previous) if previous else item
            else:
                candidates[key] = self._merge_items(previous, item)
        return list(candidates.values())

    @staticmethod
    def _merge_items(first: ContextItem, second: ContextItem) -> ContextItem:
        preferred = first if first.priority >= second.priority else second
        return ContextItem(
            document_id=preferred.document_id or first.document_id or second.document_id,
            document_title=preferred.document_title or first.document_title or second.document_title,
            document_number=preferred.document_number or first.document_number or second.document_number,
            article=preferred.article or first.article or second.article,
            clause=preferred.clause or first.clause or second.clause,
            point=preferred.point or first.point or second.point,
            validity_status=preferred.validity_status or first.validity_status or second.validity_status,
            retrieval_score=max(first.retrieval_score, second.retrieval_score),
            retrieval_source=tuple(sorted(set(first.retrieval_source + second.retrieval_source))),
            text=preferred.text,
            priority=max(first.priority, second.priority),
            citation=preferred.citation,
        )

    def _fit_item(self, item: ContextItem, character_budget: int, number: int) -> ContextItem | None:
        if character_budget <= 0:
            return None
        if len(item.render(number)) <= character_budget:
            return item
        fixed = item.render(number).replace(item.text, "")
        text_budget = character_budget - len(fixed)
        if text_budget <= 20:
            return None
        text = self._truncate_at_boundary(item.text, text_budget)
        if not text:
            return None
        return ContextItem(**{**item.__dict__, "text": text, "truncated": True})

    def _fit_to_tokens(
        self,
        item: ContextItem,
        character_budget: int,
        token_budget: int,
        number: int,
    ) -> ContextItem | None:
        if token_budget <= 0:
            return None
        estimated_characters = min(character_budget, token_budget * 4)
        return self._fit_item(item, estimated_characters, number)

    @staticmethod
    def _truncate_at_boundary(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        boundary = max(text.rfind("\n\n", 0, limit), text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if boundary < 1:
            return ""
        return text[:boundary].rstrip() + " [...]"

    def _extract_structure(
        self,
        chunk: Any,
        document: Mapping[str, Any],
    ) -> tuple[str | None, str | None, str | None]:
        articles = chunk.metadata.get("articles", document.get("articles"))
        if isinstance(articles, str):
            articles = [value.strip() for value in articles.split(",") if value.strip()]
        first_article = str(articles[0]) if articles else None
        match = self._structure_pattern.search(chunk.text)
        if match:
            point, clause, article = match.groups()
            return article.lower(), clause.lower() if clause else None, point.casefold() if point else None
        return first_article, None, None

    @staticmethod
    def _first(
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
        *keys: str,
    ) -> Any:
        for mapping in (primary, secondary):
            for key in keys:
                if mapping.get(key) is not None:
                    return mapping[key]
        return None

    @staticmethod
    def _citation(title, number, article, clause, point) -> str:
        document = title or number or "Unknown document"
        location = " ".join(
            value
            for value in (
                f"Điều {article}" if article else None,
                f"khoản {clause}" if clause else None,
                f"điểm {point}" if point else None,
            )
            if value
        )
        return f"{document}{' - ' + location if location else ''}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return (len(text) + 3) // 4 if text else 0