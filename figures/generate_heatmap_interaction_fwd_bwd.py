#!/usr/bin/env python3
"""
generate_heatmap_interaction_fwd_bwd_v2.py
==========================================
Generates the defense interaction matrix heatmap from an aggregated.json
file (the actual output format of p5_interaction_matrix.py /
p5_plot_interaction_matrix.py), with forward and backward metrics shown
in clearly separated blocks.

Input:
  --json (default: ./results/interaction_eps16/aggregated.json)
  An aggregated.json file with the schema:
    {
      "agg_raw": { "C0": {metric: [mean, std], ...}, "C1": {...}, ... },
      "agg_scr": { "C0": {metric: [mean, std], ...}, "C1": {...}, ... }
    }
  We use "agg_scr" (the score = Delta vs C0 baseline).

Output:
  --figures_dir/heatmap_interaction_fwd_bwd.png + .pdf

Config-label mapping (verified from p5_interaction_matrix.py):
  C0 = No defense                (fl_r100e1)            — baseline, all Deltas = 0
  C1 = AT only                   (fat_base)
  C2 = CoordMedian only          (coord_only)
  C3 = DP only                   (dp_only)
  C4 = AT + CoordMedian          (at_coord)
  C5 = Full stack                (fat_def_eps*)         — the headline operational regime
  C6 = CoordMedian + DP          (coord_dp)
  C7 = AT + DP                   (at_dp)

The C0 row is omitted from the heatmap (all Deltas are zero by definition).

Metric mapping:
  Forward-side  : utility_fwd, evasion_fwd
  Backward-side : utility_bwd, evasion_bwd
  Task-independent : byzantine_b_bwd (no-def), byzantine_c_bwd (+CoordMed),
                     mia_protection, dlg_mse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   10,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# Config labels (verified from p5_interaction_matrix.py)
CONFIG_LABELS = {
    "C0": "No defense",
    "C1": "AT only",
    "C2": "CoordMedian only",
    "C3": "DP only",
    "C4": "AT + CoordMedian",
    "C5": "Full stack",
    "C6": "CoordMedian + DP",
    "C7": "AT + DP",
}

# Row order in the heatmap (C0 omitted because all Deltas = 0)
ROW_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Column mapping: (display_label, json_key, block, sign_convention)
# sign_convention:
#   "as_is"           — Δ is already "positive = better" (e.g. evasion accuracy under attack)
#   "as_is"           — for utility, positive Δ = better
#   "as_is"           — for DLG MSE, positive Δ = better (more noise = more protection)
#   "as_is"           — for byzantine_b_*: positive Δ = better (higher accuracy under attack)
COLUMNS = [
    # (label, json_key, block)
    ("Utility\nfwd",            "utility_fwd",         "fwd"),
    ("Evasion\nfwd",            "evasion_fwd",         "fwd"),
    ("Utility\nbwd",            "utility_bwd",         "bwd"),
    ("Evasion\nbwd",            "evasion_bwd",         "bwd"),
    ("Byzantine\n(no def)",     "byzantine_b_bwd",     "shared"),
    ("Byzantine\n(+CoordMed)",  "byzantine_c_bwd",     "shared"),
    ("MIA\nprotection",         "mia_protection",      "shared"),
    ("DLG\nprotection",         "dlg_mse",             "shared"),
]

BLOCK_NAMES = {
    "fwd":    "Forward-side",
    "bwd":    "Backward-side",
    "shared": "Cross-task",
}


def load_data(json_path: Path):
    if not json_path.exists():
        print(f"ERROR: aggregated.json not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    with json_path.open("r") as f:
        data = json.load(f)
    if "agg_scr" not in data:
        print(f"ERROR: 'agg_scr' missing in {json_path}", file=sys.stderr)
        sys.exit(1)
    return data["agg_scr"]


def make_heatmap(agg_scr, out_base: Path, title_suffix: str = ""):
    rows = ROW_ORDER
    n_rows = len(rows)
    n_cols = len(COLUMNS)

    # Build value and std matrices
    M = np.full((n_rows, n_cols), np.nan)
    S = np.full((n_rows, n_cols), np.nan)
    for i, cfg in enumerate(rows):
        if cfg not in agg_scr:
            continue
        cdata = agg_scr[cfg]
        for j, (_, key, _) in enumerate(COLUMNS):
            if key in cdata:
                val = cdata[key]
                if isinstance(val, list) and len(val) >= 2:
                    M[i, j] = val[0]
                    S[i, j] = val[1]

    # Color scale: symmetric, based on max abs value (excluding NaN)
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.1
    if vmax == 0:
        vmax = 0.1

    # Figure size proportional to grid
    fig_w = 1.1 * n_cols + 2.3
    fig_h = 0.95 * n_rows + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            s = S[i, j]
            if not np.isfinite(v):
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#dddddd", edgecolor="white",
                    linewidth=0.5, zorder=2,
                ))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="#555555", zorder=3)
                continue
            txt = f"{v:+.3f}"
            if np.isfinite(s) and s > 0:
                txt += f"\n±{s:.3f}"
            is_strong = abs(v) > 0.5 * vmax
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5,
                    color="white" if is_strong else "black",
                    fontweight="bold" if is_strong else "normal",
                    zorder=3)

    # Ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[0] for c in COLUMNS], fontsize=9.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([CONFIG_LABELS[r] for r in rows])

    # Block dividers and headers
    blocks = [c[2] for c in COLUMNS]
    boundaries = []
    for j in range(1, n_cols):
        if blocks[j] != blocks[j - 1]:
            boundaries.append(j)
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", linewidth=3.0, zorder=4)

    # Block header labels
    unique_blocks = []
    seen = set()
    for b in blocks:
        if b not in seen:
            unique_blocks.append(b)
            seen.add(b)
    block_centers = []
    start = 0
    for end in boundaries + [n_cols]:
        block_centers.append((start + end - 1) / 2)
        start = end
    for c, b in zip(block_centers, unique_blocks):
        ax.text(c, -0.85, BLOCK_NAMES[b],
                ha="center", va="bottom", fontsize=11.5,
                fontweight="bold")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    title = ("Defense interaction matrix --- "
             r"$\Delta$ vs no-defense baseline | "
             "green = helps | red = unintended harm")
    ax.set_title(title, fontsize=11, pad=38)
    if title_suffix:
        # Place suffix as a separate annotation above the block headers
        # but below the main title, so it does not collide with either.
        fig.text(0.5, 0.945, title_suffix,
                 ha="center", va="top", fontsize=10, style="italic",
                 color="#555555")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label=r"$\Delta$ score")
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(str(out_base) + ".png")
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)
    print(f"Wrote: {out_base}.png")
    print(f"Wrote: {out_base}.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json", type=Path,
        default=Path("./results/interaction_eps16/aggregated.json"),
        help="Path to aggregated.json (default: results/interaction_eps16/aggregated.json)",
    )
    p.add_argument(
        "--figures_dir", type=Path, default=Path("./figures"),
    )
    p.add_argument(
        "--title_suffix", type=str, default="",
        help='Optional sub-title text (e.g., "at epsilon = 16, R=100, K=3")',
    )
    args = p.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    agg_scr = load_data(args.json)
    make_heatmap(
        agg_scr,
        args.figures_dir / "heatmap_interaction_fwd_bwd",
        title_suffix=args.title_suffix,
    )


if __name__ == "__main__":
    main()
