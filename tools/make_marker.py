#!/usr/bin/env python3
"""Print-ready docking marker: ArUco DICT_4X4_50 id 7, 100 mm, as PNG (300 dpi) and PDF.

    python tools/make_marker.py --out-dir markers
    # -> markers/aruco_4x4_50_id7_100mm.png and .pdf

Layout: the marker (outer edge of its black border = --size-mm) inside a white quiet zone of
--quiet-mm on every side, a light grey cutting line around the quiet zone, a label, and a 50 mm
scale bar to check the print with a ruler. Print at 100 % ("actual size", no fit-to-page).

The PDF is vector with a page sized in millimetres, so the marker is exact. The PNG is rendered
at --dpi with that resolution stored in the file; a 100 mm marker is 1181.1 px at 300 dpi, so
its edges land within half a pixel (0.04 mm) of the true size. After writing, the PNG is
detected again with the robot's own detector (jbl_nodes.aruco_utils) and the id and size are
checked.

Python 3.8 compatible; works with both OpenCV ArUco APIs (<= 4.6 and >= 4.7) through
jbl_nodes.aruco_utils.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "jbl_nodes"))
from jbl_nodes import aruco_utils  # noqa: E402

MM_PER_INCH = 25.4
MARGIN_MM = 5.0
LABEL_BLOCK_MM = 30.0
SCALE_BAR_MM = 50.0
CUT_LINE_GREY = 0.75


def marker_cells(marker_id: int, dictionary_name: str = "DICT_4X4_50") -> np.ndarray:
    """(n+2)x(n+2) bool array, True = black cell, including the one-cell black border."""
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = aruco_utils.get_dictionary(dictionary_id)
    count = dictionary.bytesList.shape[0]
    if not 0 <= marker_id < count:
        raise ValueError("%s has ids 0..%d, got %d" % (dictionary_name, count - 1, marker_id))
    bits = dictionary.markerSize
    img = aruco_utils.generate_marker(marker_id, bits + 2, dictionary_id=dictionary_id, border_bits=1)
    return np.asarray(img) < 128


def page_size_mm(size_mm: float, quiet_mm: float, dpi: int) -> Tuple[float, float]:
    """Page size rounded up to whole pixels at ``dpi``, so 1 mm maps to exactly dpi/25.4 px."""
    px_per_mm = dpi / MM_PER_INCH
    w = size_mm + 2.0 * quiet_mm + 2.0 * MARGIN_MM
    h = w + LABEL_BLOCK_MM
    return math.ceil(w * px_per_mm) / px_per_mm, math.ceil(h * px_per_mm) / px_per_mm


def _cells_path(cells: np.ndarray, x0: float, y0: float, cell_mm: float):
    """One compound path of all black cells, filled as a union (no seams between cells)."""
    from matplotlib.path import Path as MplPath

    verts: List[Tuple[float, float]] = []
    codes: List[int] = []
    for r, c in zip(*np.nonzero(cells)):
        x, y = x0 + c * cell_mm, y0 + r * cell_mm
        verts += [(x, y), (x + cell_mm, y), (x + cell_mm, y + cell_mm), (x, y + cell_mm), (x, y)]
        codes += [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def build_figure(marker_id: int, dictionary_name: str, size_mm: float, quiet_mm: float, dpi: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch, Rectangle

    cells = marker_cells(marker_id, dictionary_name)
    w_mm, h_mm = page_size_mm(size_mm, quiet_mm, dpi)
    fig = Figure(figsize=(w_mm / MM_PER_INCH, h_mm / MM_PER_INCH), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, w_mm)
    ax.set_ylim(h_mm, 0.0)  # millimetres, y down like the printed page
    ax.axis("off")

    zone = (MARGIN_MM, MARGIN_MM)
    zone_mm = size_mm + 2.0 * quiet_mm
    ax.add_patch(Rectangle(zone, zone_mm, zone_mm, fill=False, edgecolor=str(CUT_LINE_GREY),
                           linewidth=0.5, linestyle=(0, (4, 4))))
    x0 = y0 = MARGIN_MM + quiet_mm
    ax.add_patch(PathPatch(_cells_path(cells, x0, y0, size_mm / cells.shape[0]),
                           facecolor="black", edgecolor="none", antialiased=False))

    text_y = MARGIN_MM + zone_mm + 6.0
    ax.text(x0, text_y, "ArUco %s  id %d  -  %g mm black edge to black edge" % (dictionary_name, marker_id, size_mm),
            fontsize=9, va="center", ha="left")
    ax.text(x0, text_y + 5.5, "Print at 100 %% (actual size, no fit-to-page); keep the %g mm white border."
            % quiet_mm, fontsize=7, va="center", ha="left", color="0.25")
    bar_y = text_y + 11.0
    ax.add_patch(Rectangle((x0, bar_y), SCALE_BAR_MM, 2.0, facecolor="black", edgecolor="none", antialiased=False))
    for k in range(0, int(SCALE_BAR_MM) + 1, 10):
        ax.add_patch(Rectangle((x0 + k - 0.15, bar_y - 1.5), 0.3, 1.5, facecolor="black", edgecolor="none"))
    ax.text(x0 + SCALE_BAR_MM + 3.0, bar_y + 1.0, "%g mm" % SCALE_BAR_MM, fontsize=7, va="center", ha="left")
    return fig


def check_png(path: Path, marker_id: int, dictionary_name: str, size_mm: float, dpi: int) -> float:
    """Detect the marker in the written PNG; returns its measured side in mm."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cannot read back %s" % path)
    detections = aruco_utils.MarkerDetector(getattr(cv2.aruco, dictionary_name)).detect(img)
    if [d.marker_id for d in detections] != [marker_id]:
        raise RuntimeError("expected exactly marker %d in %s, detected %s" % (
            marker_id, path, [d.marker_id for d in detections]))
    q = detections[0].corners
    side_px = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))
    side_mm = side_px * MM_PER_INCH / dpi
    # Detected corners sit on the outer pixel edge to within about a pixel.
    if abs(side_mm - size_mm) > 2.0 * MM_PER_INCH / dpi:
        raise RuntimeError("marker in %s measures %.2f mm, expected %g mm" % (path, side_mm, size_mm))
    return side_mm


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--id", type=int, default=aruco_utils.DEFAULT_MARKER_ID)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--size-mm", type=float, default=100.0, help="outer edge of the black border")
    parser.add_argument("--quiet-mm", type=float, default=10.0, help="white border around the marker")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--name", default=None, help="file stem (default aruco_<dict>_id<id>_<size>mm)")
    args = parser.parse_args(argv)

    stem = args.name or "aruco_%s_id%d_%gmm" % (args.dictionary.replace("DICT_", "").lower(), args.id, args.size_mm)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.id, args.dictionary, args.size_mm, args.quiet_mm, args.dpi)
    png, pdf = args.out_dir / (stem + ".png"), args.out_dir / (stem + ".pdf")
    fig.savefig(str(png), dpi=args.dpi, facecolor="white")
    fig.savefig(str(pdf), facecolor="white")
    side_mm = check_png(png, args.id, args.dictionary, args.size_mm, args.dpi)
    w_mm, h_mm = page_size_mm(args.size_mm, args.quiet_mm, args.dpi)
    print("wrote %s and %s (page %.1f x %.1f mm; marker detected as id %d, %.2f mm in the PNG)"
          % (png, pdf, w_mm, h_mm, args.id, side_mm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
