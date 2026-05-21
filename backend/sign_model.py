import csv
import os
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


EXPECTED_FEATURE_COUNT = 88
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "sign_classifier_v1.joblib",
)


@dataclass
class TrainingArtifacts:
    model: RandomForestClassifier
    labels: list[str]
    feature_count: int
    accuracy: float
    report: str
    confusion: list[list[int]]
    train_samples: int
    test_samples: int
    model_path: str


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _coerce_feature_row(row: Iterable[str]) -> np.ndarray:
    values = [float(value) for value in row]
    features = np.asarray(values, dtype=np.float32)
    if features.shape[0] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {features.shape[0]}"
        )
    return features


def _load_labeled_csv(csv_path: str) -> tuple[list[np.ndarray], list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue

            stripped = [cell.strip() for cell in row if cell is not None]
            if not any(stripped):
                continue

            if row_index == 1 and any(not _is_number(cell) for cell in stripped[1:]):
                continue

            if len(stripped) == EXPECTED_FEATURE_COUNT + 1 and not _is_number(stripped[0]):
                label = stripped[0]
                vector = _coerce_feature_row(stripped[1:])
            elif len(stripped) == EXPECTED_FEATURE_COUNT + 1 and not _is_number(stripped[-1]):
                label = stripped[-1]
                vector = _coerce_feature_row(stripped[:-1])
            elif len(stripped) == EXPECTED_FEATURE_COUNT and _is_number(stripped[0]):
                label = os.path.splitext(os.path.basename(csv_path))[0]
                vector = _coerce_feature_row(stripped)
            else:
                raise ValueError(
                    f"{csv_path}:{row_index} has {len(stripped)} columns; "
                    f"expected {EXPECTED_FEATURE_COUNT} features plus an optional label."
                )

            labels.append(label)
            features.append(vector)

    return features, labels


def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    csv_files: list[str] = []
    if os.path.isdir(dataset_path):
        for root, _, files in os.walk(dataset_path):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, filename))
    elif dataset_path.lower().endswith(".csv"):
        csv_files.append(dataset_path)
    else:
        raise ValueError("Dataset path must be a CSV file or a directory of CSV files.")

    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    for csv_path in sorted(csv_files):
        features, labels = _load_labeled_csv(csv_path)
        all_features.extend(features)
        all_labels.extend(labels)

    if not all_features:
        raise ValueError("Dataset is empty after loading.")

    return np.vstack(all_features), np.asarray(all_labels)


def train_model(
    dataset_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 1,
) -> TrainingArtifacts:
    features, labels = load_dataset(dataset_path)

    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2:
        raise ValueError("Training requires at least 2 distinct labels.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    confusion = confusion_matrix(y_test, predictions, labels=unique_labels).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": unique_labels,
            "feature_count": EXPECTED_FEATURE_COUNT,
        },
        model_path,
    )

    return TrainingArtifacts(
        model=model,
        labels=unique_labels,
        feature_count=EXPECTED_FEATURE_COUNT,
        accuracy=accuracy,
        report=report,
        confusion=confusion,
        train_samples=len(x_train),
        test_samples=len(x_test),
        model_path=model_path,
    )


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> dict | None:
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_sign(keypoints: np.ndarray, model_bundle: dict) -> tuple[str, float]:
    vector = np.asarray(keypoints, dtype=np.float32).reshape(1, -1)
    feature_count = model_bundle.get("feature_count", EXPECTED_FEATURE_COUNT)
    if vector.shape[1] != feature_count:
        raise ValueError(f"Expected {feature_count} features, got {vector.shape[1]}")

    model: RandomForestClassifier = model_bundle["model"]
    prediction = model.predict(vector)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(np.max(probabilities))

    return str(prediction), confidence
