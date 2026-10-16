"""scripts/generate_aruco_tag.py — Generate printable ArUco Tag ID 0 (Original ArUco Format).

Generates a high-resolution, printable image and A4 document for ArUco Marker ID 0
from the Original ArUco dictionary (DICT_ARUCO_ORIGINAL) with physical measurement scale.

Usage:
    python scripts/generate_aruco_tag.py --marker_id 0 --size_mm 50 --output data/aruco_tag_0.png
    python scripts/generate_aruco_tag.py --marker_id 0 --size_mm 80 --output data/aruco_tag_0_80mm.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_aruco_marker_image(
    marker_id: int = 0,
    dictionary_name: str = "DICT_ARUCO_ORIGINAL",
    pixel_size: int = 1000,
    border_bits: int = 1,
) -> np.ndarray:
    """Generate high-resolution image array of ArUco marker."""
    dict_id = cv2.aruco.DICT_ARUCO_ORIGINAL
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # In modern OpenCV, generateImageMarker is available under cv2.aruco
    if hasattr(cv2.aruco, "generateImageMarker"):
        tag_img = cv2.aruco.generateImageMarker(dictionary, marker_id, pixel_size, borderBits=border_bits)
    else:
        tag_img = cv2.aruco.drawMarker(dictionary, marker_id, pixel_size, borderBits=border_bits)

    return tag_img


def create_printable_tag_sheet(
    marker_id: int = 0,
    size_mm: float = 50.0,
    output_path: str = "data/aruco_tag_0.png",
) -> Path:
    """Create a formatted printable sheet with target dimensions and ruler check."""
    tag_img = generate_aruco_marker_image(marker_id=marker_id, pixel_size=1200)

    # Create figure with physical A4 aspect ratio or square card
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    ax.imshow(tag_img, cmap="gray", extent=[-size_mm / 2, size_mm / 2, -size_mm / 2, size_mm / 2])
    ax.set_xlim(-size_mm * 0.9, size_mm * 0.9)
    ax.set_ylim(-size_mm * 1.1, size_mm * 0.9)
    ax.axis("off")

    # Add descriptive label and ruler calibration guide
    title_str = f"ArUco Tag ID {marker_id} (Original ArUco Format)\nTarget Physical Size: {size_mm:.1f} mm x {size_mm:.1f} mm"
    ax.text(0, size_mm * 0.72, title_str, ha="center", va="center", fontsize=11, fontweight="bold")

    # Draw a 50mm ruler line to verify print scaling
    ruler_y = -size_mm * 0.8
    ruler_len = min(50.0, size_mm)
    ax.plot([-ruler_len / 2, ruler_len / 2], [ruler_y, ruler_y], "k-", lw=2)
    ax.plot([-ruler_len / 2, -ruler_len / 2], [ruler_y - 2, ruler_y + 2], "k-", lw=2)
    ax.plot([ruler_len / 2, ruler_len / 2], [ruler_y - 2, ruler_y + 2], "k-", lw=2)
    ax.text(0, ruler_y - 5, f"<--- {ruler_len:.0f} mm (Verify with ruler when printed) --->", ha="center", va="top", fontsize=9)

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also save raw marker PNG for direct usage / digital display
    raw_path = out_p.parent / f"aruco_tag_{marker_id}_raw.png"
    cv2.imwrite(str(raw_path), tag_img)

    return out_p


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate printable ArUco Tag ID 0 (Original Format).")
    parser.add_argument("--marker_id", type=int, default=0, help="ArUco marker ID (default: 0).")
    parser.add_argument("--size_mm", type=float, default=50.0, help="Physical side length in millimetres (e.g. 50 or 80).")
    parser.add_argument("--output", type=str, default="data/aruco_tag_0_original.png", help="Output path for printable sheet.")
    args = parser.parse_args()

    out_file = create_printable_tag_sheet(marker_id=args.marker_id, size_mm=args.size_mm, output_path=args.output)
    print(f"Generated printable ArUco tag sheet: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
