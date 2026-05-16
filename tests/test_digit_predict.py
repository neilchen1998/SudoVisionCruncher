import os
import pytest

from pathlib import Path

from src.digit_predict import run_digit_prediction_pipeline

def test_run_digit_prediction_pipeline():
    """
    Tests run_digit_prediction_pipeline function
    """

    # Get the project directory (root)
    ROOT_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = ROOT_DIR / "models" / "digit_recognition_model.keras"
    IMAGE_PATH = ROOT_DIR / "data" / "nine.jpg"

    # Check the files exist
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    assert os.path.exists(IMAGE_PATH), f"Image not found at {IMAGE_PATH}"

    digit, conf = run_digit_prediction_pipeline(MODEL_PATH, IMAGE_PATH)

    # Check the digit value
    assert digit == 9

    # Check the confidence score
    assert conf > 90.0
    assert conf <= 100.0
