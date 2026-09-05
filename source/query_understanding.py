"""Dependency-light query understanding for Vietnamese legal questions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Protocol, Sequence


INTENTS = frozenset(
    {
        "legal_information",
        "article_lookup",
        "document_lookup",
        "amendment_lookup",
        "validity_lookup",
        "comparison",
        "unknown",
    }
)


@dataclass(frozen=True)
class QueryEntity:
    entity_type: str
    text: str
    normalized_value: str


@dataclass(frozen=True)
class ArticleReference:
    article: str
    clause: str | None = None
    point: str | None = None
    raw_text: str = ""

    @property
    def normalized_value(self) -> str:
        parts = [f"điều {self.article}"]
        if self.clause:
            parts.insert(0, f"khoản {self.clause}")
        if self.point:
            parts.insert(0, f"điểm {self.point}")
        return " ".join(parts)


@dataclass(frozen=True)
class StructuredQuery:
    original_query: str
    intent: str
    entities: tuple[QueryEntity, ...] = field(default_factory=tuple)
    normalized_entities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    legal_terms: tuple[str, ...] = field(default_factory=tuple)
    article_references: tuple[ArticleReference, ...] = field(default_factory=tuple)
    document_identifiers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unsupported intent: {self.intent}")

    @property
    def articles(self) -> tuple[ArticleReference, ...]:
        """Compatibility alias for callers that use the shorter field name."""

        return self.article_references

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "entities": [
                {
                    "entity_type": entity.entity_type,
                    "text": entity.text,
                    "normalized_value": entity.normalized_value,
                }
                for entity in self.entities
            ],
            "normalized_entities": {
                key: list(values) for key, values in self.normalized_entities.items()
            },
            "legal_terms": list(self.legal_terms),
            "article_references": [
                {
                    "article": reference.article,
                    "clause": reference.clause,
                    "point": reference.point,
                    "raw_text": reference.raw_text,
                }
                for reference in self.article_references
            ],
            "document_identifiers": list(self.document_identifiers),
        }


class QueryUnderstandingBackend(Protocol):
    def understand(self, question: str) -> StructuredQuery:
        ...


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _normalize_document_number(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", value.strip().upper())


class RuleBasedQueryUnderstanding:
    """Deterministic baseline backend for replacing with an LLM/NER later."""

    _reference_pattern = re.compile(
        r"(?:(?:điểm)\s+([a-zđ])\s+)?"
        r"(?:(?:khoản)\s+(\d+[a-z]?|[ivx]+)\s+)?"
        r"(?:điều)\s+(\d+[a-z]?)",
        re.IGNORECASE,
    )
    _document_number_pattern = re.compile(
        r"\b\d{1,4}\s*/\s*\d{2,4}\s*/\s*[A-ZĐÀ-Ỹ][A-ZĐÀ-Ỹ0-9-]*\b",
        re.IGNORECASE,
    )
    _date_pattern = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})\b",
        re.IGNORECASE,
    )
    _law_name_pattern = re.compile(
        r"\b(?:luật|bộ luật|nghị định|thông tư|quyết định)\s+"
        r".*?(?=\s+(?:quy định|về|có hiệu lực)\b|[,;?\n]|$)",
        re.IGNORECASE,
    )
    _organization_pattern = re.compile(
        r"\b(?:quốc hội|chính phủ|bộ|sở|ủy ban nhân dân|tòa án nhân dân)"
        r"[^,;?\n]*",
        re.IGNORECASE,
    )
    _legal_term_pattern = re.compile(
        r"(?:hiệu lực|thẩm quyền|xử phạt|hợp đồng|quyền và nghĩa vụ|"
        r"trách nhiệm|đối tượng áp dụng|phạm vi điều chỉnh|điều kiện|"
        r"sửa đổi|thay thế|bãi bỏ|hủy bỏ)",
        re.IGNORECASE,
    )

    def understand(self, question: str) -> StructuredQuery:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        original_query = question.strip()
        normalized_query = _normalize_text(original_query)
        references = self._extract_references(original_query)
        document_identifiers = self._extract_document_identifiers(original_query)
        entities = self._extract_entities(original_query, references)
        legal_terms = self._extract_legal_terms(normalized_query)
        normalized_entities = self._group_normalized_entities(entities)
        intent = self._infer_intent(
            normalized_query,
            references,
            document_identifiers,
        )
        return StructuredQuery(
            original_query=original_query,
            intent=intent,
            entities=tuple(entities),
            normalized_entities=normalized_entities,
            legal_terms=tuple(legal_terms),
            article_references=tuple(references),
            document_identifiers=tuple(document_identifiers),
        )

    def _extract_references(self, question: str) -> list[ArticleReference]:
        references = []
        for match in self._reference_pattern.finditer(question):
            point, clause, article = match.groups()
            references.append(
                ArticleReference(
                    article=article.lower(),
                    clause=clause.lower() if clause else None,
                    point=point.casefold() if point else None,
                    raw_text=match.group(0),
                )
            )
        return references

    def _extract_document_identifiers(self, question: str) -> list[str]:
        identifiers = []
        for match in self._document_number_pattern.finditer(question):
            value = _normalize_document_number(match.group(0))
            if value not in identifiers:
                identifiers.append(value)
        return identifiers

    def _extract_entities(
        self,
        question: str,
        references: Sequence[ArticleReference],
    ) -> list[QueryEntity]:
        entities: list[QueryEntity] = []
        for reference in references:
            if reference.point:
                entities.append(QueryEntity("point", reference.point, reference.point))
            if reference.clause:
                entities.append(QueryEntity("clause", reference.clause, reference.clause))
            entities.append(QueryEntity("article", reference.raw_text, reference.article))
        for match in self._document_number_pattern.finditer(question):
            value = _normalize_document_number(match.group(0))
            entities.append(QueryEntity("document_number", match.group(0), value))
        for pattern, entity_type in (
            (self._law_name_pattern, "law_name"),
            (self._date_pattern, "date"),
            (self._organization_pattern, "organization"),
        ):
            for match in pattern.finditer(question):
                text = match.group(0).strip(" .,;?")
                entities.append(QueryEntity(entity_type, text, _normalize_text(text)))
        for term in self._extract_legal_terms(_normalize_text(question)):
            entities.append(QueryEntity("legal_concept", term, term))
        return self._deduplicate_entities(entities)

    @staticmethod
    def _deduplicate_entities(entities: Sequence[QueryEntity]) -> list[QueryEntity]:
        result = []
        seen = set()
        for entity in entities:
            key = (entity.entity_type, entity.normalized_value)
            if key not in seen:
                seen.add(key)
                result.append(entity)
        return result

    def _extract_legal_terms(self, normalized_query: str) -> list[str]:
        terms = []
        for match in self._legal_term_pattern.finditer(normalized_query):
            term = _normalize_text(match.group(0))
            if term not in terms:
                terms.append(term)
        return terms

    @staticmethod
    def _group_normalized_entities(
        entities: Sequence[QueryEntity],
    ) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for entity in entities:
            grouped.setdefault(entity.entity_type, [])
            if entity.normalized_value not in grouped[entity.entity_type]:
                grouped[entity.entity_type].append(entity.normalized_value)
        return {key: tuple(values) for key, values in grouped.items()}

    @staticmethod
    def _infer_intent(
        normalized_query: str,
        references: Sequence[ArticleReference],
        document_identifiers: Sequence[str],
    ) -> str:
        if any(term in normalized_query for term in ("so sánh", "khác nhau", "giống nhau")):
            return "comparison"
        if any(term in normalized_query for term in ("hiệu lực", "còn hiệu lực", "hết hiệu lực")):
            return "validity_lookup"
        if any(term in normalized_query for term in ("sửa đổi", "thay thế", "bãi bỏ", "hủy bỏ")):
            return "amendment_lookup"
        if references:
            return "article_lookup"
        if document_identifiers or any(
            term in normalized_query
            for term in ("văn bản nào", "nghị định nào", "luật nào", "thông tư nào")
        ):
            return "document_lookup"
        if any(
            term in normalized_query
            for term in (
                "luật",
                "điều",
                "khoản",
                "điểm",
                "văn bản",
                "quy định",
                "pháp luật",
                "quyền",
                "nghĩa vụ",
                "trách nhiệm",
            )
        ):
            return "legal_information"
        return "unknown"


class QueryUnderstandingService:
    """Stable application interface delegating to a replaceable backend."""

    def __init__(self, backend: QueryUnderstandingBackend | None = None) -> None:
        self.backend = backend or RuleBasedQueryUnderstanding()

    def understand(self, question: str) -> StructuredQuery:
        return self.backend.understand(question)