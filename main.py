import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pathlib import Path

from src.digit_predict import load_model
from src.parse_sudoku_board import flatten_board, parse_sudoku_board
from src.profiler import PipelineProfiler
from src.render_sudoku_solution import *
from src.solver import solve_sudoku

def print_board(board: list[list], width:int = 3):
    """
    Prints the Sudoku board

    Args:
        board: The Sudoku board
        width: The width of each digit in character
    """

    # The top border
    print("+" + ("-" * (width + 2) + "+") * len(board[0]))

    for row in board:
        print("|" + "|".join(f" {str(cell).center(width)} " for cell in row) + "|")
        print("+" + ("-" * (width + 2) + "+") * len(board[0]))

def main():

    # Set the default model path
    ROOT_DIR = Path.cwd()
    MODEL_PATH = ROOT_DIR / "models" / "digit_recognition_model_TMNIST.keras"

    # Create a parser
    parser = argparse.ArgumentParser(prog="SudoVisionCruncher", description="Hello, user:")

    # Positional argument(s) (require by default)
    parser.add_argument("image_path", type=Path, help="The path of the Sudoku image")

    # Optional argument(s)
    parser.add_argument("-m", "--model-path", type=Path, default=MODEL_PATH, help="The file path of the OCR model")
    parser.add_argument("-V", "--verbose", action="store_true", default=False, help="Print the pipeline profile summary")

    # Parse the arguments
    args = parser.parse_args()

    # Check the paths exist
    if not args.image_path.exists():
        parser.error(f"Image not found: {args.image_path}")

    if not args.model_path.exists():
        parser.error(f"Model not found: {args.model_path}")

    profiler = PipelineProfiler()

    # Load the model
    model = profiler.profile(
        "Load model",
        load_model,
        str(args.model_path)
    )

    # Import the image and turn into grey scale
    img = profiler.profile(
        "Read image",
        cv2.imread,
        str(args.image_path)
    )

    flatten_img, M = profiler.profile(
        "Flatten board",
        flatten_board,
        img
    )

    # Parse the given Sudoku board
    board, empty_positions = profiler.profile(
        "OCR",
        parse_sudoku_board,
        flatten_img,
        model
    )

    # Solve the Sudoku board
    solved_board = profiler.profile(
        "Solve Sudoku",
        solve_sudoku,
        board
    )

    # Render the overlay
    overlay = profiler.profile(
        "Render overlay",
        render_solution_overlay,
        solved_board,
        empty_positions
    )

    # Overlay the solution on the original image
    result = profiler.profile(
        "Project overlay",
        overlay_solution_on_board,
        img,
        overlay,
        M
    )

    # Print the summary report if 'verbose' is provided by the user
    if args.verbose:
        profiler.report()

    # Display the image
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) # NOTE: OpenCV uses BGR but matplotlib uses RGB
    plt.axis('off')
    plt.title("Solved Sudoku")
    plt.show()

    return

if __name__ == "__main__":

    main()
