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

# creates the CSV columns
def build_feature_header() -> list[str]:
    """Create stable CSV column names for two hands plus the final label."""
    columns: list[str] = []

    for hand_name in ("left", "right"):
        for landmark_index in range(LANDMARKS_PER_HAND):
            for axis in ("x", "y", "z"):
                columns.append(f"{hand_name}_lm{landmark_index}_{axis}")

    columns.append("label")
    return columns

# subtracts wrist x/y/z from every landmark
def normalize_hand_landmarks(hand_landmarks: object) -> list[float]:
    """
    Convert one MediaPipe hand into wrist-relative x/y/z coordinates.
    """
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

#  turns MediaPipe output into 126 numbers
def extract_features(results: object) -> tuple[list[float], int]:
    """
    Return a fixed-width feature row and the number of valid detected hands.

    Feature layout is always:
        [left hand 63 values] + [right hand 63 values]

    Missing hands are padded with zeros so every CSV row has exactly
    126 numeric features before the label.
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

# creates data/hand_data.csv with headers
def ensure_csv_exists(csv_path: Path) -> None:
    """Create the dataset CSV with a header if it does not already exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(build_feature_header())

#  saves one labeled training row
def append_sample(csv_path: Path, features: Iterable[float], label_id: int) -> None:
    """Append one normalized training sample to the dataset CSV."""
    row = list(features)

    if len(row) != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, got {len(row)}."
        )

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row + [label_id])