"""generate_checkerboard_pdf.py — Generates an A4 calibration chessboard pattern.

Produces a 10x7 squares (9x6 internal corners) chessboard with 25mm squares,
properly centered on an A4 page with technical annotations for camera calibration.

Usage:
    python generate_checkerboard_pdf.py [--output damier_calibration_A4.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_checkerboard_pdf(
    output_filename: str | Path = "damier_calibration_A4.pdf",
    inner_corners_x: int = 9,
    inner_corners_y: int = 6,
    square_size_mm: float = 25.0,
) -> Path:
    """Generate and save an A4 chessboard calibration PDF.

    Args:
        output_filename: Output PDF destination.
        inner_corners_x: Number of internal corners in X (default 9).
        inner_corners_y: Number of internal corners in Y (default 6).
        square_size_mm: Square edge length in millimeters (default 25.0 mm).

    Returns:
        Path to the saved PDF file.
    """
    squares_x = inner_corners_x + 1  # 10 squares
    squares_y = inner_corners_y + 1  # 7 squares

    a4_width_mm = 297.0
    a4_height_mm = 210.0

    mm_to_inch = 1.0 / 25.4
    fig_width = a4_width_mm * mm_to_inch
    fig_height = a4_height_mm * mm_to_inch

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    grid_w = squares_x * square_size_mm
    grid_h = squares_y * square_size_mm
    offset_x = (a4_width_mm - grid_w) / 2.0
    offset_y = (a4_height_mm - grid_h) / 2.0

    # Draw black squares
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 1:
                rect = patches.Rectangle(
                    (offset_x + col * square_size_mm, offset_y + row * square_size_mm),
                    square_size_mm,
                    square_size_mm,
                    facecolor="black",
                    edgecolor="none",
                )
                ax.add_patch(rect)

    info_text = (
        f"OpenCV Checkerboard | Coins internes : {inner_corners_x}x{inner_corners_y} | "
        f"Taille de case : {square_size_mm:.1f} mm"
    )
    ax.text(
        a4_width_mm / 2.0,
        offset_y - 8,
        info_text,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="sans-serif",
        color="gray",
    )

    ax.set_xlim(0, a4_width_mm)
    ax.set_ylim(0, a4_height_mm)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out_path = Path(output_filename)
    plt.savefig(
        str(out_path),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    print(f"Damier généré avec succès : {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate A4 calibration checkerboard PDF.")
    parser.add_argument("--output", default="damier_calibration_A4.pdf", help="Output PDF file path.")
    parser.add_argument("--inner_x", type=int, default=9, help="Inner corners X.")
    parser.add_argument("--inner_y", type=int, default=6, help="Inner corners Y.")
    parser.add_argument("--square_mm", type=float, default=25.0, help="Square size in mm.")
    args = parser.parse_args()

    generate_checkerboard_pdf(
        output_filename=args.output,
        inner_corners_x=args.inner_x,
        inner_corners_y=args.inner_y,
        square_size_mm=args.square_mm,
    )
