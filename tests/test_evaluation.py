import json

from source.evaluation import (
    EvaluationExample,
    evaluate_generation,
    evaluate_retrieval,
    load_evaluation_dataset,
    retrieval_metrics,
    write_report,
)


def test_retrieval_metrics():
    metrics = retrieval_metrics(
        [{"document_id": "doc-2"}, {"document_id": "doc-1"}],
        ["doc-1"],
        k=2,
    )

    assert metrics["recall@2"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["hit_rate@2"] == 1.0


def test_retrieval_evaluator_supports_modes_and_summary():
    dataset = [EvaluationExample("q", ("doc-1",), ())]
    report = evaluate_retrieval(dataset, lambda question: [{"document_id": "doc-1"}], mode="vector")

    assert report.summary["recall@5"] == 1.0
    assert report.rows[0]["mode"] == "vector"


def test_generation_metrics_and_outputs(tmp_path):
    dataset = [EvaluationExample("Điều 5?", ground_truth_answer="Điều 5 nói về hợp đồng")]
    report = evaluate_generation(
        dataset,
        lambda question: {
            "answer": "Điều 5 nói về hợp đồng",
            "citations": [{"is_valid": True, "is_trusted": True}],
        },
        {"Điều 5?": "Điều 5 nói về hợp đồng"},
    )
    output = tmp_path / "report"
    write_report(report, output)

    assert report.summary["faithfulness"] == 1.0
    assert report.summary["citation_correctness"] == 1.0
    assert output.with_suffix(".json").exists()
    assert output.with_suffix(".csv").exists()
    assert output.with_suffix(".summary.txt").exists()


def test_json_and_csv_dataset_loading(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"question": "q", "ground_truth_documents": ["d1"]}], ensure_ascii=False), encoding="utf-8")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("question,ground_truth_documents\nq,d1|d2\n", encoding="utf-8")

    assert load_evaluation_dataset(json_path)[0].ground_truth_documents == ("d1",)
    assert load_evaluation_dataset(csv_path)[0].ground_truth_documents == ("d1", "d2")