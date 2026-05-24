"""
Real-time hand gesture recognition using the trained Random Forest model.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Import collection utilities
from collect_data import (
    DEFAULT_MODEL_PATH,
    MAX_HANDS,
    TOTAL_FEATURES,
    create_hand_landmarker,
    draw_hand_landmarks,
    ensure_hand_landmarker_model,
    extract_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.pkl"


def load_gesture_model(model_path: Path) -> dict:
    """Load the trained model package. Auto-trains using synthetic/existing data if missing."""
    if not model_path.exists() or model_path.stat().st_size == 0:
        print(f"Model file {model_path} not found. Running training script first...")
        train_script = PROJECT_ROOT / "src" / "train.py"
        try:
            subprocess.run([sys.executable, str(train_script)], check=True)
        except subprocess.CalledProcessError as err:
            raise RuntimeError("Failed to auto-train the gesture classifier model.") from err

    with model_path.open("rb") as file:
        model_package = pickle.load(file)
    
    return model_package


def draw_prediction(
    frame: object,
    label: str,
    confidence: float,
    hands_detected: int,
) -> None:
    """Draw prediction status and HUD on the camera frame."""
    # Top HUD background
    cv2.rectangle(frame, (0, 0), (320, 110), (20, 20, 20), -1)
    # Border
    cv2.rectangle(frame, (0, 0), (320, 110), (30, 220, 30), 1)

    # Status labels
    cv2.putText(
        frame,
        f"Detected Hands: {hands_detected}/{MAX_HANDS}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if hands_detected > 0:
        # Show predicted label with high-visibility styling
        cv2.putText(
            frame,
            f"Gesture: {label}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )
        # Show confidence/probability
        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 220, 30),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Gesture: None",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Place hand in frame",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    """Read command line options."""
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognition.")
    parser.add_argument("--camera", default=0, type=int, help="OpenCV camera index. Defaults to 0.")
    parser.add_argument(
        "--model-task",
        default=str(DEFAULT_MODEL_PATH),
        help="MediaPipe Hand Landmarker .task model path.",
    )
    parser.add_argument(
        "--model-pkl",
        default=str(MODEL_PATH),
        help="Path to trained gesture classifier package (.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load classifiers
    model_pkl_path = Path(args.model_pkl)
    model_package = load_gesture_model(model_pkl_path)
    
    clf = model_package["classifier"]
    id_to_label = model_package["id_to_label"]
    
    task_model_path = Path(args.model_task)
    ensure_hand_landmarker_model(task_model_path)
    
    # Open camera
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Check webcam connection or try --camera 1."
        )
        
    frame_timestamp_ms = 0
    print("Starting gesture recognition window. Press 'Q' or 'Esc' to exit.")
    
    try:
        with create_hand_landmarker(task_model_path) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Warning: camera frame dropped.")
                    continue
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_timestamp_ms += 1
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                hands_detected = 0
                label = "None"
                confidence = 0.0
                
                if results.hand_landmarks:
                    features, hands_detected = extract_features(results)
                    
                    if hands_detected > 0:
                        # Reshape features for model prediction
                        features_arr = np.array(features).reshape(1, -1)
                        
                        # Predict label and get probabilities
                        pred_id = clf.predict(features_arr)[0]
                        pred_probs = clf.predict_proba(features_arr)[0]
                        
                        label = id_to_label.get(pred_id, "Unknown")
                        # Find the index of the predicted ID in classifier classes
                        class_idx = np.where(clf.classes_ == pred_id)[0]
                        if len(class_idx) > 0:
                            confidence = pred_probs[class_idx[0]]
                    
                    for hand_landmarks in results.hand_landmarks:
                        draw_hand_landmarks(frame, hand_landmarks)
                        
                draw_prediction(frame, label, confidence, hands_detected)
                cv2.imshow("DigiGestures - Real-time Classifier", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Gesture recognition finished.")


if __name__ == "__main__":
    main()
