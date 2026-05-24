"""
Collect normalized MediaPipe hand landmarks for GestureSense-AI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hand_data.csv"

MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
TOTAL_FEATURES = MAX_HANDS * FEATURES_PER_HAND

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "123456789"

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(LETTERS + DIGITS)
}

ID_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_ID.items()
}


def build_feature_header() -> list[str]:
    """Create stable CSV column names for two hands plus the final label."""
    columns: list[str] = []

    for hand_name in ("left", "right"):
        for landmark_index in range(LANDMARKS_PER_HAND):
            for axis in ("x", "y", "z"):
                columns.append(f"{hand_name}_lm{landmark_index}_{axis}")

    columns.append("label")
    return columns


def normalize_hand_landmarks(hand_landmarks: object) -> list[float]:
    """Convert one MediaPipe hand into wrist-relative x/y/z coordinates."""
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]

    normalized: list[float] = []

    for landmark in landmarks:
        normalized.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            ]
        )

    return normalized


def extract_features(results: object) -> tuple[list[float], int]:
    """
    Return a fixed-width feature row and the number of valid detected hands.

    Feature layout is always:
        [left hand 63 values] + [right hand 63 values]
    """
    empty_hand = [0.0] * FEATURES_PER_HAND
    hand_slots = {
        "Left": empty_hand.copy(),
        "Right": empty_hand.copy(),
    }

    hand_landmarks = results.multi_hand_landmarks or []
    handedness_values = results.multi_handedness or []

    for landmarks, handedness in zip(
        hand_landmarks[:MAX_HANDS],
        handedness_values[:MAX_HANDS],
    ):
        handedness_label = handedness.classification[0].label

        if handedness_label not in hand_slots:
            continue

        hand_slots[handedness_label] = normalize_hand_landmarks(landmarks)

    features = hand_slots["Left"] + hand_slots["Right"]
    detected_count = min(len(hand_landmarks), MAX_HANDS)

    return features, detected_count


def ensure_csv_exists(csv_path: Path) -> None:
    """Create the dataset CSV with a header if it does not already exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(build_feature_header())


def append_sample(csv_path: Path, features: Iterable[float], label_id: int) -> None:
    """Append one normalized training sample to the dataset CSV."""
    row = list(features)

    if len(row) != TOTAL_FEATURES:
        raise ValueError(f"Expected {TOTAL_FEATURES} features, got {len(row)}.")

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row + [label_id])


def parse_args() -> argparse.Namespace:
    """Read command-line options for the collector."""
    parser = argparse.ArgumentParser(
        description="Collect normalized hand landmark samples for gesture recognition."
    )
    parser.add_argument("--label", required=True, type=str, help="Gesture label to collect. Use A-Z or 1-9.")
    parser.add_argument("--camera", default=0, type=int, help="OpenCV camera index. Defaults to 0.")
    parser.add_argument("--output", default=str(DATA_PATH), type=str, help="CSV output path.")
    return parser.parse_args()


def draw_status(
    frame: object,
    label_text: str,
    label_id: int,
    saved_count: int,
    detected_hands: int,
) -> None:
    """Draw capture status text onto the webcam frame."""
    status_lines = [
        f"Collecting: {label_text} -> {label_id}",
        f"Saved this run: {saved_count}",
        f"Hands detected: {detected_hands}/{MAX_HANDS}",
        "Press C to capture | Q/Esc to quit",
    ]

    y_position = 30
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )
        y_position += 30


def main() -> None:
    """Run the webcam data collection loop."""
    args = parse_args()
    label_text = args.label.strip().upper()

    if label_text not in LABEL_TO_ID:
        valid_labels = ", ".join(LETTERS + DIGITS)
        raise ValueError(f"Invalid label '{args.label}'. Use one of: {valid_labels}")

    label_id = LABEL_TO_ID[label_text]
    output_path = Path(args.output)
    ensure_csv_exists(output_path)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Check webcam permissions or try --camera 1."
        )

    saved_count = 0
    latest_features = [0.0] * TOTAL_FEATURES
    latest_detected_hands = 0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Warning: camera frame dropped; trying next frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.multi_hand_landmarks:
                    latest_features, latest_detected_hands = extract_features(results)

                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                else:
                    latest_features = [0.0] * TOTAL_FEATURES
                    latest_detected_hands = 0
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                draw_status(
                    frame,
                    label_text,
                    label_id,
                    saved_count,
                    latest_detected_hands,
                )
                cv2.imshow("GestureSense-AI Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                if key == ord("c"):
                    if latest_detected_hands == 0:
                        print("No hand detected. Sample was not saved.")
                        continue

                    append_sample(output_path, latest_features, label_id)
                    saved_count += 1
                    print(f"Saved sample {saved_count} for label {label_text}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Collection finished. Saved {saved_count} samples to {output_path}.")


if __name__ == "__main__":
    main()
