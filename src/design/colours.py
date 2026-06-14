from difflib import get_close_matches
import webcolors

CSS_COLOURS = list(webcolors.names())

def bgr(name: str) -> tuple[int, int, int]:
    """
    Coverts a given CSS colour name to OpenCV BGR colour tuple

    Args:
        name: The given CSS name

    Returns:
        tuple[int, int, int]: The colour as BGR tuple for OpenCV

    Raises:
        ValueError: If the given colour name is not recognized
    """

    try:
        rgb = webcolors.name_to_rgb(name)

        return (rgb.blue, rgb.green, rgb.red)

    except ValueError:
        suggestion = get_close_matches(name.lower(), CSS_COLOURS, n=1)

        # Show the suggestion(s) if any
        if suggestion:
            raise ValueError(f"Unknown colour: {name}. Did you mean '{suggestion[0]}'?") from None  # suppresses the original exception

        # No suggestion can be found
        raise ValueError(f"Unknown colour: '{name}'.") from None
