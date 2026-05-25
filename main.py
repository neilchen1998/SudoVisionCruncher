import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pathlib import Path

from src.digit_predict import load_model
from src.parse_sudoku_board import flatten_board, parse_sudoku_board
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
    parser.add_argument("image_path", type=str, help="The path of the Sudoku image")

    # Optional argument(s)
    parser.add_argument("-m", "--model-path", type=str, default=MODEL_PATH, help="The file path of the OCR model")

    # Parse the arguments
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Import the image and turn into grey scale
    img = cv2.imread(args.image_path)
    flatten_img, M = flatten_board(img)

    board, empty_positions = parse_sudoku_board(flatten_img, model)

    solved_board = solve_sudoku(board)

    overlay = draw_solution_overlay(
        solved_board,
        empty_positions
    )

    result = project_overlay_back(
        img,
        overlay,
        M
    )

    # Display the image
    plt.imshow(result)
    plt.axis('off')
    plt.title("Solved Sudoku")
    plt.show()

    return

if __name__ == "__main__":

    main()
