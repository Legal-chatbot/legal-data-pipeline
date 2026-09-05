"""Evaluation utilities for Vietnamese legal RAG components and pipelines."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    ground_truth_documents: tuple[str, ...] = field(default_factory=tuple)
    ground_truth_articles: tuple[str, ...] = field(default_factory=tuple)
    ground_truth_answer: str = ""
    ground_truth_intent: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    task: str
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "rows": list(self.rows), "summary": dict(self.summary)}


def load_evaluation_dataset(path: str | Path) -> list[EvaluationExample]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("examples", payload) if isinstance(payload, dict) else payload
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError("evaluation dataset must be JSON or CSV")
    if not isinstance(records, list):
        raise ValueError("evaluation dataset must contain a list of examples")
    return [_example_from_record(record) for record in records]


def _example_from_record(record: Mapping[str, Any]) -> EvaluationExample:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("each evaluation example needs a question")
    return EvaluationExample(
        question=question,
        ground_truth_documents=tuple(_as_list(record.get("ground_truth_documents"))),
        ground_truth_articles=tuple(_as_list(record.get("ground_truth_articles"))),
        ground_truth_answer=str(record.get("ground_truth_answer", "") or ""),
        ground_truth_intent=record.get("ground_truth_intent") or None,
    )


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in value.split("|") if item.strip()]
    return [str(value).strip()]


def retrieval_metrics(
    retrieved: Sequence[Any],
    ground_truth_documents: Sequence[str],
    ground_truth_articles: Sequence[str] = (),
    k: int = 5,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be greater than zero")
    expected_documents = {_normalize(value) for value in ground_truth_documents}
    expected_articles = {_normalize_article(value) for value in ground_truth_articles}
    ranked = [_retrieved_record(item) for item in retrieved[:k]]
    relevant = [
        record
        for record in ranked
        if _is_relevant(record, expected_documents, expected_articles)
    ]
    relevant_keys = {
        key
        for record in relevant
        for key in (record.get("document_id"), record.get("article"))
        if key
    }
    expected_count = len(expected_documents or expected_articles)
    recall = len(relevant_keys & (expected_documents or expected_articles)) / expected_count if expected_count else 0.0
    precision = len(relevant) / k
    reciprocal_rank = 1.0 / (next((index + 1 for index, record in enumerate(ranked) if record in relevant), 0) or float("inf"))
    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        "mrr": reciprocal_rank,
        f"hit_rate@{k}": 1.0 if relevant else 0.0,
    }


def evaluate_retrieval(
    dataset: Sequence[EvaluationExample],
    retriever: Any,
    k: int = 5,
    mode: str = "hybrid",
) -> EvaluationReport:
    rows = []
    for example in dataset:
        result = _call_component(retriever, example.question)
        retrieved = _retrieved_items(result)
        metrics = retrieval_metrics(
            retrieved,
            example.ground_truth_documents,
            example.ground_truth_articles,
            k,
        )
        rows.append({"question": example.question, "mode": mode, "retrieved_count": len(retrieved), **metrics})
    return _aggregate_report("retrieval", rows)


def evaluate_query_understanding(dataset: Sequence[EvaluationExample], analyzer: Any) -> EvaluationReport:
    rows = []
    for example in dataset:
        result = _call_component(analyzer, example.question)
        intent = getattr(result, "intent", None)
        rows.append(
            {
                "question": example.question,
                "intent_correct": bool(example.ground_truth_intent and intent == example.ground_truth_intent),
                "predicted_intent": intent,
            }
        )
    return _aggregate_report("query_understanding", rows)


def evaluate_generation(
    dataset: Sequence[EvaluationExample],
    generator: Any,
    contexts: Mapping[str, str] | None = None,
) -> EvaluationReport:
    rows = []
    for example in dataset:
        output = _call_component(generator, example.question)
        answer = _output_answer(output)
        context = (contexts or {}).get(example.question, "")
        rows.append(
            {
                "question": example.question,
                "faithfulness": _text_overlap(answer, context),
                "context_relevance": _text_overlap(example.question, context),
                "answer_relevance": _text_overlap(example.question, answer),
                "answer_reference_overlap": _text_overlap(example.ground_truth_answer, answer),
                "citation_correctness": _citation_score(output),
            }
        )
    return _aggregate_report("generation", rows)


def evaluate_full_pipeline(
    dataset: Sequence[EvaluationExample],
    pipeline: Any,
    k: int = 5,
) -> EvaluationReport:
    rows = []
    for example in dataset:
        output = _call_component(pipeline, example.question)
        retrieval = getattr(output, "retrieval_result", output)
        retrieved = _retrieved_items(retrieval)
        retrieval_result = retrieval_metrics(
            retrieved,
            example.ground_truth_documents,
            example.ground_truth_articles,
            k,
        )
        answer = _output_answer(output)
        context = _output_context(output)
        rows.append(
            {
                "question": example.question,
                "answer": answer,
                **retrieval_result,
                "faithfulness": _text_overlap(answer, context),
                "context_relevance": _text_overlap(example.question, context),
                "answer_relevance": _text_overlap(example.question, answer),
                "citation_correctness": _citation_score(output),
            }
        )
    return _aggregate_report("full_pipeline", rows)


def write_report(report: EvaluationReport, output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if report.rows:
        fieldnames = sorted({key for row in report.rows for key in row})
        with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report.rows)
    prefix.with_suffix(".summary.txt").write_text(
        "Evaluation summary\n"
        f"Task: {report.task}\n"
        f"Examples: {len(report.rows)}\n"
        + "\n".join(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in report.summary.items()),
        encoding="utf-8",
    )


def _aggregate_report(task: str, rows: Sequence[Mapping[str, Any]]) -> EvaluationReport:
    numeric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    summary = {
        key: sum(float(row[key]) for row in rows if key in row) / len(rows)
        for key in numeric_keys
    } if rows else {}
    return EvaluationReport(task, tuple(rows), summary)


def _call_component(component: Any, question: str) -> Any:
    if callable(component):
        return component(question)
    if hasattr(component, "retrieve"):
        return component.retrieve(question)
    if hasattr(component, "understand"):
        return component.understand(question)
    if hasattr(component, "answer"):
        return component.answer(question)
    raise TypeError("evaluation component must be callable or expose retrieve/understand/answer")


def _retrieved_items(result: Any) -> list[Any]:
    if result is None:
        return []
    if hasattr(result, "chunks"):
        documents = tuple(getattr(result, "documents", ()))
        scores = tuple(getattr(result, "scores", ()))
        items = []
        for index, chunk in enumerate(result.chunks):
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            document = documents[index] if index < len(documents) else metadata
            item = dict(document)
            item["document_id"] = item.get("document_id", item.get("id", item.get("doc_id")))
            item["score"] = scores[index] if index < len(scores) else None
            item["article"] = metadata.get("article")
            item["articles"] = metadata.get("articles", item.get("articles"))
            items.append(item)
        return items
    return list(result) if isinstance(result, (list, tuple)) else []


def _retrieved_record(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        record = dict(item)
    else:
        metadata = dict(getattr(item, "metadata", {}) or {})
        record = metadata
        record.update({"document_id": getattr(item, "document_id", record.get("document_id"))})
    document_id = record.get("document_id", record.get("id", record.get("doc_id")))
    article = record.get("article")
    articles = record.get("articles")
    if article is None and articles:
        article = articles[0] if isinstance(articles, (list, tuple)) else str(articles).split(",")[0]
    return {"document_id": _normalize(document_id), "article": _normalize_article(article)}


def _is_relevant(record, documents, articles) -> bool:
    return bool((record.get("document_id") and record["document_id"] in documents) or (record.get("article") and record["article"] in articles))


def _normalize(value: Any) -> str | None:
    return str(value).strip().casefold() if value is not None and str(value).strip() else None


def _normalize_article(value: Any) -> str | None:
    normalized = _normalize(value)
    return normalized.removeprefix("điều ") if normalized else None


def _text_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[\wÀ-ỹĐđ]+", (left or "").casefold()))
    right_tokens = set(re.findall(r"[\wÀ-ỹĐđ]+", (right or "").casefold()))
    return len(left_tokens & right_tokens) / len(left_tokens) if left_tokens else 0.0


def _citation_score(output: Any) -> float:
    citations = tuple(
        output.get("citations", ()) if isinstance(output, Mapping) else getattr(output, "citations", ())
    )
    if not citations:
        return 0.0
    return sum(
        bool(
            (citation.get("is_valid") and citation.get("is_trusted"))
            if isinstance(citation, Mapping)
            else getattr(citation, "is_valid", False) and getattr(citation, "is_trusted", False)
        )
        for citation in citations
    ) / len(citations)


def _output_answer(output: Any) -> str:
    if isinstance(output, Mapping):
        return str(output.get("answer", "") or "")
    return str(getattr(output, "answer", output) or "")


def _output_context(output: Any) -> str:
    if isinstance(output, Mapping):
        return str(output.get("context", "") or "")
    return str(getattr(output, "context", "") or "")