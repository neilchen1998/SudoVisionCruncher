import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pathlib import Path

from src.digit_predict import load_model
from src.parse_sudoku_board import flatten_board, parse_sudoku_board

def main():

    # Set the default model path
    ROOT_DIR = Path.cwd()
    MODEL_PATH = ROOT_DIR / "models" / "digit_recognition_model_TMNIST.keras"

    # Create a parser
    parser = argparse.ArgumentParser(prog="SudoVisionCruncher", description="Hello, user:")

    # Positional argument(s) (require by default)
    parser.add_argument("image_path", type=str, help="The path of the Sudoku image")

    # Optional argument(s)
    parser.add_argument("-m", "--model-path", type=str, default=MODEL_PATH, help="The file path of the OCR model")

    # Parse the arguments
    args = parser.parse_args()

    # Import the image and turn into grey scale
    img = cv2.imread(args.image_path)
    flatten_img = flatten_board(img)

    board = parse_sudoku_board(flatten_img, args.model_path)

    print(board)

    return

if __name__ == "__main__":

    main()
