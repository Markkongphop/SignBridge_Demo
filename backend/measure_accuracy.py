import argparse
import csv
import os

import cv2
import mediapipe as mp

from holistic_data_extract import extract_keypoints
from sign_model import DEFAULT_MODEL_PATH, load_trained_model, predict_sign


RESULTS_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "accuracy_results.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real-time webcam sign classification accuracy."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--prediction-count",
        type=int,
        default=30,
        help="Number of predictions to record after pressing 's'. Default: 30",
    )
    parser.add_argument(
        "--results-csv",
        default=RESULTS_CSV_PATH,
        help=f"CSV file used to append accuracy results. Default: {RESULTS_CSV_PATH}",
    )
    return parser


def append_accuracy_result(
    csv_path: str,
    target_sign: str,
    correct_predictions: int,
    total_predictions: int,
    accuracy_percentage: float,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                ["target_sign", "correct_predictions", "total_predictions", "accuracy_percentage"]
            )
        writer.writerow(
            [
                target_sign,
                correct_predictions,
                total_predictions,
                f"{accuracy_percentage:.2f}",
            ]
        )


def measure_accuracy(
    target_sign: str,
    model_path: str = DEFAULT_MODEL_PATH,
    camera_index: int = 0,
    prediction_count: int = 30,
    results_csv: str = RESULTS_CSV_PATH,
) -> float:
    if not target_sign.strip():
        raise ValueError("Target sign cannot be empty.")

    model_bundle = load_trained_model(model_path)
    if model_bundle is None:
        raise FileNotFoundError(f"Model not found: {model_path}")

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    is_recording = False
    total_predictions = 0
    correct_predictions = 0
    last_prediction = ""
    normalized_target_sign = target_sign.strip()

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)

                status_text = "Press 's' to start accuracy recording | 'q' to quit"
                status_color = (0, 255, 0)

                if is_recording:
                    status_text = (
                        f"Recording {total_predictions}/{prediction_count} | Target: {normalized_target_sign}"
                    )
                    status_color = (0, 255, 255)

                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        keypoints = extract_keypoints(results)
                        predicted_sign, confidence = predict_sign(keypoints, model_bundle)
                        last_prediction = predicted_sign
                        total_predictions += 1

                        if predicted_sign == normalized_target_sign:
                            correct_predictions += 1

                        print(
                            f"Prediction {total_predictions}/{prediction_count}: "
                            f"{predicted_sign} | confidence={confidence:.2f} | "
                            f"match={predicted_sign == normalized_target_sign}"
                        )

                        if total_predictions >= prediction_count:
                            break
                    else:
                        status_text = (
                            f"Recording {total_predictions}/{prediction_count} | No hands detected"
                        )
                        status_color = (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Target: {normalized_target_sign}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    status_text,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Last prediction: {last_prediction or '-'}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Correct: {correct_predictions}/{prediction_count}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SignBridge - Accuracy Measurement", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s") and not is_recording:
                    is_recording = True
                    total_predictions = 0
                    correct_predictions = 0
                    last_prediction = ""
                    print(
                        f"Started accuracy recording for target sign: {normalized_target_sign}"
                    )
                elif key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if total_predictions == 0:
        print("No predictions were recorded.")
        return 0.0

    accuracy_percentage = (correct_predictions / total_predictions) * 100.0
    append_accuracy_result(
        csv_path=results_csv,
        target_sign=normalized_target_sign,
        correct_predictions=correct_predictions,
        total_predictions=total_predictions,
        accuracy_percentage=accuracy_percentage,
    )

    print("\nFinal Accuracy Result")
    print(f"Target sign: {normalized_target_sign}")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
    print(f"Saved results to: {results_csv}")
    return accuracy_percentage


def main() -> None:
    args = build_parser().parse_args()
    target_sign = input("Enter the target sign you are going to perform: ").strip()
    measure_accuracy(
        target_sign=target_sign,
        model_path=args.model,
        camera_index=args.camera_index,
        prediction_count=args.prediction_count,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
