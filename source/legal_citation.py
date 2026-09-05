"""Metadata-derived legal citations and answer citation validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SourceDocument:
    document_id: str | None
    title: str | None
    document_number: str | None
    validity_status: str | None


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str | None
    text: str
    document: SourceDocument


@dataclass(frozen=True, eq=False)
class Citation:
    citation_id: str
    label: str
    source_document: SourceDocument
    source_chunk: SourceChunk
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    validity_status: str | None = None
    is_valid: bool = True
    is_trusted: bool = False
    used_in_answer: bool = False
    invalid_reason: str | None = None

    def __str__(self) -> str:
        return self.label

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Citation):
            return self.citation_id == other.citation_id
        if isinstance(other, str):
            return self.label == other
        return NotImplemented


@dataclass(frozen=True)
class CitationValidation:
    citations: tuple[Citation, ...]
    invalid_references: tuple[str, ...]
    missing_citation: bool


class LegalCitationSystem:
    """Create citations only from retrieved context and validate LLM markers."""

    _marker_pattern = re.compile(r"\[(C\d+)\]")

    def from_context(self, context: Any) -> tuple[Citation, ...]:
        items = tuple(getattr(context, "items", ()))
        citations = []
        seen = set()
        for index, item in enumerate(items, start=1):
            document = SourceDocument(
                document_id=self._value(item, "document_id"),
                title=self._value(item, "document_title"),
                document_number=self._value(item, "document_number"),
                validity_status=self._value(item, "validity_status"),
            )
            chunk = SourceChunk(
                chunk_id=self._value(item, "chunk_id"),
                text=str(self._value(item, "text") or ""),
                document=document,
            )
            key = (chunk.chunk_id, chunk.text, document.document_id)
            if key in seen:
                continue
            seen.add(key)
            article = self._value(item, "article")
            clause = self._value(item, "clause")
            point = self._value(item, "point")
            label = self._value(item, "citation") or self._label(
                document,
                article,
                clause,
                point,
            )
            citations.append(
                Citation(
                    citation_id=f"C{len(citations) + 1}",
                    label=label,
                    source_document=document,
                    source_chunk=chunk,
                    article=article,
                    clause=clause,
                    point=point,
                    validity_status=document.validity_status,
                )
            )
        return tuple(citations)

    def validate(self, answer: str, citations: Sequence[Citation]) -> CitationValidation:
        markers = self._marker_pattern.findall(answer)
        known = {citation.citation_id: citation for citation in citations}
        invalid = tuple(dict.fromkeys(marker for marker in markers if marker not in known))
        used = set(markers).intersection(known)
        validated = tuple(
            replace(
                citation,
                is_trusted=citation.citation_id in used,
                used_in_answer=citation.citation_id in used,
                is_valid=citation.citation_id not in invalid,
            )
            for citation in citations
        )
        invalid_citations = tuple(
            Citation(
                citation_id=marker,
                label=marker,
                source_document=SourceDocument(None, None, None, None),
                source_chunk=SourceChunk(None, "", SourceDocument(None, None, None, None)),
                is_valid=False,
                invalid_reason="citation marker was not present in retrieved context",
            )
            for marker in invalid
        )
        return CitationValidation(
            citations=validated + invalid_citations,
            invalid_references=invalid,
            missing_citation=bool(citations) and not bool(markers),
        )

    @staticmethod
    def _value(item: Any, name: str) -> str | None:
        value = getattr(item, name, None)
        if value is None and hasattr(item, "metadata"):
            value = item.metadata.get(name)
        return str(value) if value is not None else None

    @staticmethod
    def _label(document: SourceDocument, article, clause, point) -> str:
        name = document.title or document.document_number or document.document_id or "Unknown document"
        location = " ".join(
            value
            for value in (
                f"Điều {article}" if article else None,
                f"khoản {clause}" if clause else None,
                f"điểm {point}" if point else None,
            )
            if value
        )
        return f"{name}{' - ' + location if location else ''}"