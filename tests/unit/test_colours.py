import pytest

from src.colours import bgr

def test_bgr_with_know_colours():
    """
    Test bgr function with know colours.
    """

    # black = RGB (0, 0, 0) -> BGR (0, 0, 0)
    assert bgr ("black") == (0, 0, 0)

    # red = RGB (255, 0, 0) -> BGR (0, 0, 255)
    assert bgr ("red") == (0, 0, 255)

    # green = RGB (0, 128, 0) -> BGR (0, 128, 0)
    assert bgr ("green") == (0, 128, 0)

    # blue = RGB (0, 0, 255) -> BGR (255, 0, 0)
    assert bgr ("blue") == (255, 0, 0)

    # white = RGB (255, 255, 255) -> BGR (255, 255, 255)
    assert bgr ("white") == (255, 255, 255)

def test_bgr_with_unknow_colours():

    green_incorrect_str = "gren"

    with pytest.raises(ValueError) as excinfo:
        bgr (green_incorrect_str)

    assert str(excinfo.value) == (
        f"Unknown colour: {green_incorrect_str}. Did you mean 'green'?"
    )

    purple_incorrect_str = "purpl"

    with pytest.raises(ValueError) as excinfo:
        bgr (purple_incorrect_str)

    assert str(excinfo.value) == (
        f"Unknown colour: {purple_incorrect_str}. Did you mean 'purple'?"
    )


def test_unknown_colour_without_close_match_raises_error():
    with pytest.raises(ValueError) as excinfo:
        bgr ("xyz123")

    assert str(excinfo.value) == "Unknown colour: 'xyz123'."


def test_error_message_preserves_original_input_case():
    with pytest.raises(ValueError) as excinfo:
        bgr ("GrEnn")

    message = str(excinfo.value)

    assert "GrEnn" in message
    assert "Did you mean 'green'?" in message
