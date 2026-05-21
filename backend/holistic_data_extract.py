import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def cal_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def extract_keypoints(results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        l_sh = [landmarks[11].x, landmarks[11].y]
        l_el = [landmarks[13].x, landmarks[13].y]
        l_wr = [landmarks[15].x, landmarks[15].y]

        r_sh = [landmarks[12].x, landmarks[12].y]
        r_el = [landmarks[14].x, landmarks[14].y]
        r_wr = [landmarks[16].x, landmarks[16].y]

        angle_l_elbow = cal_angle(l_sh, l_el, l_wr)
        angle_r_elbow = cal_angle(r_sh, r_el, r_wr)

        l_hip_ref = [l_sh[0], l_sh[1] + 1.0]
        r_hip_ref = [r_sh[0], r_sh[1] + 1.0]

        angle_l_shoulder = cal_angle(l_hip_ref, l_sh, l_el)
        angle_r_shoulder = cal_angle(r_hip_ref, r_sh, r_el)

        pose_angles = np.array(
            [angle_l_elbow, angle_r_elbow, angle_l_shoulder, angle_r_shoulder]
        )
    else:
        pose_angles = np.zeros(4)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[point.x, point.y] for point in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left_hand = np.zeros(21 * 2)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[point.x, point.y] for point in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        right_hand = np.zeros(21 * 2)

    return np.concatenate([pose_angles, left_hand, right_hand])


def run_data_collection(label: str, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    is_recording = False
    frames_recorded = 0

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if is_recording:
                keypoints = extract_keypoints(results)
                with open(output_path, mode="a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([label, *keypoints.tolist()])
                frames_recorded += 1
                status_text = f"REC {label} frames={frames_recorded}"
                status_color = (0, 0, 255)
            else:
                status_text = f"Label={label} | R=start S=stop Q=quit"
                status_color = (0, 255, 0)

            cv2.putText(
                image,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SignBridge - Data Collection", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("r"):
                is_recording = True
                print(f"Start recording label='{label}' -> {output_path}")
            elif key == ord("s"):
                is_recording = False
                print(f"Stop recording. Total frames written: {frames_recorded}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Collect labeled 88-keypoint sign data.")
    parser.add_argument("--label", required=True, help="Target label for this recording session.")
    parser.add_argument(
        "--output",
        default=os.path.join("backend", "data", "raw", "sign_samples.csv"),
        help="CSV path to append labeled samples.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_data_collection(label=arguments.label, output_path=arguments.output)
