"""
Train a machine learning model to classify hand gestures from normalized landmarks.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hand_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.pkl"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "123456789"
CUSTOM_WORDS = ["LOVE", "HELLO", "Fuck You", "OK", "YES", "NO"]
LABEL_TO_ID = {label: index for index, label in enumerate(list(LETTERS + DIGITS) + CUSTOM_WORDS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

# Total features expected: 2 hands * 21 landmarks * 3 coords = 126
TOTAL_FEATURES = 126


def generate_synthetic_data(num_samples_per_class: int = 50) -> pd.DataFrame:
    """
    Generate synthetic hand landmark data for a few classes to enable testing.
    This creates basic geometric configurations for gestures: A, B, C, 1, 2, 3.
    """
    print("Generating synthetic hand data for pipeline verification...")
    np.random.seed(42)
    
    # Target classes: 'A', 'B', 'C', '1', '2', '3'
    target_labels = ["A", "B", "C", "1", "2", "3"]
    target_ids = [LABEL_TO_ID[label] for label in target_labels]
    
    records = []
    
    for label_id in target_ids:
        # Create base landmarks depending on the gesture class
        # For simplicity, we create randomized patterns that differ significantly per class
        for _ in range(num_samples_per_class):
            left_hand = np.zeros(63)
            right_hand = np.zeros(63)
            
            # Populate right hand with distinct patterns
            # Wrist is always 0, 0, 0
            # Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
            
            # Baseline coordinates for a simple hand
            base_hand = []
            for finger_idx in range(5):
                # 4 landmarks per finger extending outward
                angle = finger_idx * 0.4
                for joint_idx in range(1, 5):
                    # closed or open depending on label
                    is_open = True
                    if label_id == LABEL_TO_ID["A"]:  # Closed fist
                        is_open = False
                    elif label_id == LABEL_TO_ID["1"] and finger_idx != 1:  # Only index finger open
                        is_open = False
                    elif label_id == LABEL_TO_ID["2"] and finger_idx not in (1, 2):  # Index and middle open
                        is_open = False
                    elif label_id == LABEL_TO_ID["3"] and finger_idx not in (1, 2, 3):  # Index, middle, ring open
                        is_open = False
                    
                    extension = (joint_idx * 0.1) if is_open else (joint_idx * 0.02)
                    base_hand.extend([
                        np.cos(angle) * extension + np.random.normal(0, 0.01),
                        np.sin(angle) * extension + np.random.normal(0, 0.01),
                        -joint_idx * 0.02 + np.random.normal(0, 0.01)
                    ])
            
            # Pad wrist coordinates (0, 0, 0)
            right_hand_landmarks = [0.0, 0.0, 0.0] + base_hand
            
            # Left hand stays all zeros (un-detected) for single-hand gestures
            features = left_hand.tolist() + right_hand_landmarks
            features.append(label_id)
            records.append(features)
            
    # Create column headers matching collect_data.py
    columns = []
    for hand_name in ("left", "right"):
        for landmark_index in range(21):
            for axis in ("x", "y", "z"):
                columns.append(f"{hand_name}_lm{landmark_index}_{axis}")
    columns.append("label")
    
    return pd.DataFrame(records, columns=columns)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load collected data or fallback to synthetic data if empty."""
    if not csv_path.exists() or csv_path.stat().st_size <= 2000:  # less than ~2KB is just header
        df = generate_synthetic_data()
        # Save synthetic data to CSV so it exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved synthetic dataset to {csv_path}")
        return df
        
    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"Dataset has only {len(df)} samples. Adding synthetic data to help training...")
        synthetic_df = generate_synthetic_data()
        df = pd.concat([df, synthetic_df], ignore_index=True)
        
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a gesture classification model.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to input CSV file.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to save output model (.pkl).")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    
    # Load dataset
    df = load_dataset(data_path)
    print(f"Loaded dataset with {len(df)} samples.")
    
    # Split features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Ensure shape matches
    if X.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"Features dimension mismatch. Expected {TOTAL_FEATURES}, got {X.shape[1]}")
        
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples.")
    print(f"Testing set: {X_test.shape[0]} samples.")
    
    # Train Random Forest Classifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Print classification report using label mappings
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [ID_TO_LABEL[lid] for lid in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pack model along with label mappings and model metadata
    model_package = {
        "classifier": clf,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "feature_count": TOTAL_FEATURES
    }
    
    with model_path.open("wb") as file:
        pickle.dump(model_package, file)
        
    print(f"Successfully saved model to {model_path}")


if __name__ == "__main__":
    main()
