FONT_NAME = [
    "Helvetica",
    "AppleGothic",
    "Arial",
    "Georgia",
    "Times",
]

FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Times.ttc",
]

FONT_TO_LABEL = {path : idx for idx, path in enumerate(FONT_PATHS)}

LABEL_TO_FONT_NAME = {idx : name for idx, name in enumerate(FONT_NAME)}

LABEL_TO_FONT = {idx : path for idx, path in enumerate(FONT_PATHS)}
