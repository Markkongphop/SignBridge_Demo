import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from holistic_data_extract import extract_keypoints
from sign_model import DEFAULT_MODEL_PATH, load_trained_model, predict_sign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end webcam inference latency for sign prediction."
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
        "--max-predictions",
        type=int,
        default=20,
        help="Number of successful predictions to measure before exiting. Default: 20",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("backend", "latency_results.csv"),
        help="CSV file used to append latency measurement results.",
    )
    return parser


def append_latency_result(
    output_path: str,
    tested_sign: str,
    average_latency_ms: float,
    latencies_ms: list[float],
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    file_exists = os.path.exists(output_path)
    needs_header = not file_exists or os.path.getsize(output_path) == 0

    with open(output_path, mode="a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(
                ["Date_Time", "Tested_Sign", "Average_Latency_ms", "All_20_Latencies"]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                tested_sign,
                round(average_latency_ms, 2),
                ", ".join(f"{value:.2f}" for value in latencies_ms),
            ]
        )


def main() -> None:
    args = build_parser().parse_args()
    tested_sign = input("Enter the sign you are testing for latency: ").strip()
    if not tested_sign:
        tested_sign = "unknown"

    model_bundle = load_trained_model(args.model)
    if model_bundle is None:
        raise FileNotFoundError(f"Model not found: {args.model}")

    mp_holistic = mp.solutions.holistic
    latencies_ms: list[float] = []
    successful_predictions = 0
    last_prediction = ""
    last_latency_ms = 0.0

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            start_time = time.perf_counter()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)

            if not (results.left_hand_landmarks or results.right_hand_landmarks):
                cv2.putText(
                    frame,
                    "No hands detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                keypoints = extract_keypoints(results)
                predicted_text, _confidence = predict_sign(keypoints, model_bundle)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                latencies_ms.append(elapsed_ms)
                successful_predictions += 1
                last_prediction = predicted_text
                last_latency_ms = elapsed_ms

                print(
                    f"Prediction {successful_predictions}/{args.max_predictions}: "
                    f"{predicted_text} | latency={elapsed_ms:.2f} ms"
                )

                if successful_predictions >= args.max_predictions:
                    average_latency_ms = float(np.mean(latencies_ms))
                    append_latency_result(
                        output_path=args.output,
                        tested_sign=tested_sign,
                        average_latency_ms=average_latency_ms,
                        latencies_ms=latencies_ms,
                    )
                    print(f"Latencies (ms): {[round(value, 2) for value in latencies_ms]}")
                    print(f"Average latency: {average_latency_ms:.2f} ms")
                    print(f"Results saved to: {os.path.abspath(args.output)}")
                    break

            cv2.putText(
                frame,
                f"Prediction: {last_prediction or '-'}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Latency: {last_latency_ms:.2f} ms",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Count: {successful_predictions}/{args.max_predictions}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SignBridge - Latency Measurement", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
