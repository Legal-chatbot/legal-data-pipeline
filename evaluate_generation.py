"""CLI: evaluate answer/context predictions from a JSON file."""

import argparse
import json

from source.evaluation import evaluate_generation, load_evaluation_dataset, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="evaluation/generation")
    args = parser.parse_args()
    predictions = json.loads(open(args.predictions, encoding="utf-8").read())
    by_question = {row["question"]: row for row in predictions}
    report = evaluate_generation(
        load_evaluation_dataset(args.dataset),
        lambda question: by_question.get(question, {}),
        {question: row.get("context", "") for question, row in by_question.items()},
    )
    write_report(report, args.output)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()