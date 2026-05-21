import argparse
import json
import os

from sign_model import DEFAULT_MODEL_PATH, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest sign classifier from 88-keypoint CSV data."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to one labeled CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the trained model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = train_model(
        dataset_path=args.data,
        model_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
    )

    summary = {
        "model_path": os.path.abspath(artifacts.model_path),
        "feature_count": artifacts.feature_count,
        "labels": artifacts.labels,
        "train_samples": artifacts.train_samples,
        "test_samples": artifacts.test_samples,
        "accuracy": round(artifacts.accuracy, 4),
        "confusion_matrix": artifacts.confusion,
        "classification_report": artifacts.report,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
