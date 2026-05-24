"""
Collect normalized MediaPipe hand landmarks for GestureSense-AI.
"""

from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hand_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
TOTAL_FEATURES = MAX_HANDS * FEATURES_PER_HAND

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "123456789"
CUSTOM_WORDS = ["LOVE", "HELLO", "Fuck You", "OK", "YES", "NO"]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(list(LETTERS + DIGITS) + CUSTOM_WORDS)
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


def normalize_hand_landmarks(hand_landmarks: list) -> list[float]:
    """Convert one MediaPipe hand into wrist-relative x/y/z coordinates."""
    landmarks = hand_landmarks
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

    hand_landmarks = (
        getattr(results, "multi_hand_landmarks", None)
        or getattr(results, "hand_landmarks", None)
        or []
    )
    handedness_values = (
        getattr(results, "multi_handedness", None)
        or getattr(results, "handedness", None)
        or []
    )

    for landmarks, handedness in zip(
        hand_landmarks[:MAX_HANDS],
        handedness_values[:MAX_HANDS],
    ):
        if hasattr(handedness, "classification"):
            handedness_label = handedness.classification[0].label
        else:
            handedness_label = handedness[0].category_name

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
    parser.add_argument(
        "--label", default="A", type=str,
        help="Gesture label to collect. Use A-Z, 1-9, or custom words: "
             + ", ".join(CUSTOM_WORDS),
    )
    parser.add_argument("--camera", default=0, type=int, help="OpenCV camera index. Defaults to 0.")
    parser.add_argument("--output", default=str(DATA_PATH), type=str, help="CSV output path.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        type=str,
        help="MediaPipe Hand Landmarker .task model path.",
    )
    return parser.parse_args()


def ensure_hand_landmarker_model(model_path: Path) -> None:
    """Download the MediaPipe Tasks hand model when it is not present locally."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand model to {model_path}...")

    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            "Could not download the MediaPipe hand model. "
            f"Download it manually from {HAND_LANDMARKER_URL} and save it as {model_path}."
        ) from error


def create_hand_landmarker(model_path: Path) -> object:
    """Create a MediaPipe Tasks hand landmarker for video frames."""
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(frame: object, hand_landmarks: Iterable[object]) -> None:
    """Draw MediaPipe Tasks hand landmarks with OpenCV."""
    frame_height, frame_width = frame.shape[:2]
    points = [
        (int(landmark.x * frame_width), int(landmark.y * frame_height))
        for landmark in hand_landmarks
    ]

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (30, 180, 255),
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(frame, point, 4, (30, 220, 30), -1, cv2.LINE_AA)


def draw_status(
    frame: object,
    label_text: str,
    label_id: int,
    saved_count: int,
    detected_hands: int,
) -> None:
    """Draw capture status text onto the webcam frame."""
    status_lines = [
        f"Active Label: {label_text} (ID: {label_id})",
        f"Saved samples: {saved_count}",
        f"Hands detected: {detected_hands}/{MAX_HANDS}",
        "------------------------------",
        "Press [A-Z] / [1-9] to switch label",
        "Or use --label LOVE, HELLO, etc.",
        "Press Space to capture | Esc to quit",
    ]

    # Draw dark translucent background HUD
    cv2.rectangle(frame, (5, 5), (370, 210), (20, 20, 20), -1)
    cv2.rectangle(frame, (5, 5), (370, 210), (30, 220, 30), 1)

    y_position = 30
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 220, 30),
            1,
            cv2.LINE_AA,
        )
        y_position += 25


def main() -> None:
    """Run the webcam data collection loop."""
    args = parse_args()
    label_text = args.label.strip().upper()

    if label_text not in LABEL_TO_ID:
        valid_labels = ", ".join(list(LETTERS + DIGITS) + CUSTOM_WORDS)
        raise ValueError(f"Invalid label '{args.label}'. Use one of: {valid_labels}")

    label_id = LABEL_TO_ID[label_text]
    output_path = Path(args.output)
    model_path = Path(args.model)
    ensure_csv_exists(output_path)
    ensure_hand_landmarker_model(model_path)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Check webcam permissions or try --camera 1."
        )

    saved_count = 0
    latest_features = [0.0] * TOTAL_FEATURES
    latest_detected_hands = 0
    frame_timestamp_ms = 0

    print(f"Starting data collection. Active label: {label_text}")
    print("Press Space to capture a sample.")
    print("Press any letter (A-Z) or digit (1-9) to switch active label live.")
    print(f"Custom word labels available via --label flag: {', '.join(CUSTOM_WORDS)}")
    print("Press Esc to exit.")

    try:
        with create_hand_landmarker(model_path) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Warning: camera frame dropped; trying next frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms += 1
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if results.hand_landmarks:
                    latest_features, latest_detected_hands = extract_features(results)

                    for hand_landmarks in results.hand_landmarks:
                        draw_hand_landmarks(frame, hand_landmarks)
                else:
                    latest_features = [0.0] * TOTAL_FEATURES
                    latest_detected_hands = 0
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (15, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
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
                cv2.imshow("DigiGestures Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc
                    break

                if key == ord(" "):  # Space to capture
                    if latest_detected_hands == 0:
                        print("No hand detected. Sample was not saved.")
                        continue

                    append_sample(output_path, latest_features, label_id)
                    saved_count += 1
                    print(f"Saved sample {saved_count} for label {label_text}.")
                else:
                    # Check if pressed key corresponds to a letter or number to switch active label
                    try:
                        char_key = chr(key).upper()
                        if char_key in LABEL_TO_ID:
                            label_text = char_key
                            label_id = LABEL_TO_ID[label_text]
                            print(f"Switched active label to: {label_text} (ID: {label_id})")
                    except ValueError:
                        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Collection finished. Saved {saved_count} samples to {output_path}.")


if __name__ == "__main__":
    main()
