"""Draw boxes on fundus images: annotations (solid) and, optionally, predictions (dashed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from .annotations import Box  # noqa: E402
from .labels import color_for  # noqa: E402


@dataclass
class Panel:
    title: str
    image_path: Path
    boxes: list[Box]
    predictions: list[tuple[Box, float | None]] = field(default_factory=list)


def draw_panels(panels: list[Panel], output_path: Path, columns: int = 2, zoom: bool = False, line_width: float = 1.4, dpi: int = 150) -> Path:
    """Save a grid of images with their boxes. zoom=True crops each panel to the area around its boxes."""
    if not panels:
        raise ValueError("nothing to draw")
    columns = max(1, min(columns, len(panels)))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 6 * rows), squeeze=False)

    for axis, panel in zip(axes.flat, panels):
        with Image.open(panel.image_path) as source:
            axis.imshow(source.convert("RGB"))
        for box in panel.boxes:
            axis.add_patch(Rectangle((box.x1, box.y1), box.width, box.height, fill=False,
                                     edgecolor=color_for(box.label), linewidth=line_width))
        for box, confidence in panel.predictions:
            axis.add_patch(Rectangle((box.x1, box.y1), box.width, box.height, fill=False,
                                     edgecolor=color_for(box.label), linewidth=line_width, linestyle="--"))

        all_boxes = list(panel.boxes) + [b for b, _ in panel.predictions]
        if zoom and all_boxes:
            margin = 60
            axis.set_xlim(min(b.x1 for b in all_boxes) - margin, max(b.x2 for b in all_boxes) + margin)
            axis.set_ylim(max(b.y2 for b in all_boxes) + margin, min(b.y1 for b in all_boxes) - margin)

        labels = sorted({b.label for b in all_boxes})
        handles = [Line2D([0], [0], color=color_for(label), lw=2, label=label) for label in labels]
        if panel.predictions:
            handles += [Line2D([0], [0], color="grey", lw=2, label="annotation (solid)"),
                        Line2D([0], [0], color="grey", lw=2, linestyle="--", label="prediction (dashed)")]
        title = f"{panel.title}  ({len(panel.boxes)} box{'es' if len(panel.boxes) != 1 else ''}"
        title += f", {len(panel.predictions)} predicted)" if panel.predictions else ")"
        axis.set_title(title, fontsize=9)
        if handles:
            axis.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.88)
        axis.axis("off")

    for axis in axes.flat[len(panels):]:
        axis.axis("off")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
