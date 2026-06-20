from importlib.metadata import version, metadata
from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt

from sudovisioncruncher.design.colours import bgr
from sudovisioncruncher.design.font import LABEL_TO_FONT_NAME
from sudovisioncruncher.profiler.profiler import PipelineProfiler
from sudovisioncruncher.render.render_sudoku_solution import render_solution_overlay, overlay_solution_on_board
from sudovisioncruncher.sudoku.solver import solve_sudoku
from sudovisioncruncher.vision.digit_predict import load_model
from sudovisioncruncher.vision.parse_sudoku_board import flatten_board, parse_sudoku_board

def main():

    # Set the default model path
    ROOT_DIR = Path(__file__).resolve().parents[2]
    OCR_MODEL_PATH = ROOT_DIR / "models" / "digit_recognition_model_TMNIST.keras"
    FONT_MODEL_PATH = ROOT_DIR / "models" / "font_recognition_MobileNetV2.keras"

    # Get the version from TOML
    __version__ = version("sudovisioncruncher")

    # Create a parser
    parser = argparse.ArgumentParser(
        prog="SudoVisionCruncher",
        description="",
        epilog="Example:\n  uv run -m main ./data/sudoku.png",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional argument(s) (require by default)
    parser.add_argument("image_path", type=Path, help="The path of the Sudoku image")

    # Optional argument(s)
    parser.add_argument("-m", "--ocr-model-path", type=Path, default=OCR_MODEL_PATH, help="The file path of the OCR model")
    parser.add_argument("-f", "--font-model-path", type=Path, default=FONT_MODEL_PATH, help="The file path of the font detection model")
    parser.add_argument("-V", "--verbose", action="store_true", default=False, help="Print the pipeline profile summary")
    parser.add_argument("--output", "-o", help="The output path", default=None)
    parser.add_argument("--colour", "-c", type=str, help="Overlay colour", default='red')
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Parse the arguments
    args = parser.parse_args()

    # Check the paths exist
    if not args.image_path.exists():
        parser.error(f"Image not found: {args.image_path}")

    if not args.ocr_model_path.exists():
        parser.error(f"OCR model not found: {args.ocr_model_path}")

    if not args.font_model_path.exists():
        parser.error(f"Font detection model not found: {args.font_model_path}")

    profiler = PipelineProfiler()

    # Load the model
    ocr_model = profiler.profile(
        "Load model",
        load_model,
        str(args.ocr_model_path),
        args.verbose
    )
    font_model = profiler.profile(
        "Load model",
        load_model,
        str(args.font_model_path),
        args.verbose
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
    board, empty_positions, font_label = profiler.profile(
        "OCR & Font detection",
        parse_sudoku_board,
        flatten_img,
        ocr_model,
        font_model
    )

    if args.verbose:
        print(f"Font detected: {LABEL_TO_FONT_NAME[font_label]}")

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
        empty_positions,
        font_label,
        colour=bgr(args.colour)
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # NOTE: OpenCV uses BGR but matplotlib uses RGB
    ax.set_title("Solved Sudoku", pad=2)
    ax.axis('off')
    plt.tight_layout(pad=1.0)
    plt.show()

    if args.output:
        resized_result = cv2.resize(result, (350, 350))
        cv2.imwrite(str(args.output), resized_result)

    return

if __name__ == "__main__":

    main()
