"""CLI: evaluate stored full-pipeline predictions."""

import argparse
import json

from source.evaluation import evaluate_full_pipeline, load_evaluation_dataset, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="evaluation/full_pipeline")
    args = parser.parse_args()
    predictions = json.loads(open(args.predictions, encoding="utf-8").read())
    by_question = {row["question"]: row for row in predictions}
    report = evaluate_full_pipeline(
        load_evaluation_dataset(args.dataset),
        lambda question: by_question.get(question, {}),
        args.k,
    )
    write_report(report, args.output)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()