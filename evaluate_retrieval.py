"""CLI: evaluate retrieval predictions from JSON/CSV files."""

import argparse
import json

from source.evaluation import evaluate_retrieval, load_evaluation_dataset, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True, help="JSON/CSV predictions keyed by question")
    parser.add_argument("--mode", choices=("vector", "graph", "hybrid"), default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="evaluation/retrieval")
    args = parser.parse_args()
    predictions = json.loads(open(args.predictions, encoding="utf-8").read())
    by_question = {row["question"]: row.get("retrieved", []) for row in predictions}
    report = evaluate_retrieval(load_evaluation_dataset(args.dataset), lambda question: by_question.get(question, []), args.k, args.mode)
    write_report(report, args.output)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()