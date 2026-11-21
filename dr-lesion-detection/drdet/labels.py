"""Colours used when drawing each lesion type."""

from __future__ import annotations

# Same colours as the original plotter, except soft exudate (it was a second yellow,
# easy to confuse with hard exudate). Cotton wool spots are white clinically, so white.
LABEL_COLORS = {
    "Hard Retinal Exudate (Retinal Lipid Deposit)": "#f2c14e",
    "New Vessels Elsewhere (NVE)": "#56b4e9",
    "New Vessels at Disc (NVD)": "#009e73",
    "Retinal Blot Haemorrhage": "#e76f51",
    "Retinal Dot Haemorrhage": "#cc79a7",
    "Soft Retinal Exudate (Cotton wool spot)": "#ffffff",
    "Retinal Microaneurysm": "#d55e00",
}

UNKNOWN_COLOR = "#ff00ff"  # magenta, so an unexpected label stands out


def color_for(label: str) -> str:
    return LABEL_COLORS.get(label, UNKNOWN_COLOR)
