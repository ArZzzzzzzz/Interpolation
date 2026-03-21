"""
poster/04_tile_visual.py — Schéma illustratif du découpage en tuiles avec zones d'overlap

Génère une figure claire montrant :
  - Une image 1920×1080 découpée en tuiles 256×256
  - Les zones de recouvrement (overlap) en surbrillance
  - Un zoom sur 3 tuiles adjacentes pour voir l'overlap en détail

Usage:
    cd poster
    python 04_tile_visual.py
    python 04_tile_visual.py --video "../19201080_before.mp4"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def get_patch_coords(H, W, patch_size, overlap):
    stride = patch_size - overlap
    coords = []
    y = 0
    while True:
        y0 = min(y, H - patch_size)
        x = 0
        while True:
            x0 = min(x, W - patch_size)
            coords.append((y0, x0))
            if x0 + patch_size >= W:
                break
            x += stride
        if y0 + patch_size >= H:
            break
        y += stride
    return coords


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      type=str, default=os.path.join(ROOT, "19201080_before.mp4"))
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--overlap",    type=int, default=32)
    p.add_argument("--frame_idx",  type=int, default=5)
    p.add_argument("--out_dir",    type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def make_checkerboard(H, W, tile=64):
    """Fond en damier gris clair/blanc pour simuler une frame neutre."""
    arr = np.ones((H, W, 3), dtype=np.uint8) * 220
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            if ((y // tile) + (x // tile)) % 2 == 0:
                arr[y:y+tile, x:x+tile] = 200
    return arr


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    W, H = 1920, 1080
    patch_size = args.patch_size
    overlap    = args.overlap
    stride     = patch_size - overlap

    # Charger ou créer une frame de fond
    frame_img = None
    cap = cv2.VideoCapture(args.video)
    if cap.isOpened():
        for _ in range(args.frame_idx):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Frame chargée depuis {args.video}")

    if frame_img is None:
        print("Vidéo non disponible, utilisation d'un fond synthétique.")
        frame_img = make_checkerboard(H, W)

    coords = get_patch_coords(H, W, patch_size, overlap)
    n_tiles = len(coords)
    n_x = len(set(c[1] for c in coords))
    n_y = len(set(c[0] for c in coords))

    print(f"Résolution : {W}×{H} | patch={patch_size} | overlap={overlap} | stride={stride}")
    print(f"Grille : {n_y}×{n_x} = {n_tiles} tuiles")

    # =========================================================================
    # Figure 1 — Vue d'ensemble : image avec grille de tuiles
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame_img, alpha=0.75)

    # Colormap par tuile pour montrer l'alternance
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, (y0, x0) in enumerate(coords):
        col = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x0, y0), patch_size, patch_size,
            linewidth=0.6, edgecolor=col, facecolor=col, alpha=0.08
        )
        ax.add_patch(rect)
        rect_border = patches.Rectangle(
            (x0, y0), patch_size, patch_size,
            linewidth=0.6, edgecolor="white", facecolor="none", alpha=0.6
        )
        ax.add_patch(rect_border)

    # Zone d'overlap horizontale (entre 2 tuiles adjacentes)
    if len(coords) >= 2:
        y0, x0 = coords[0]
        y1, x1 = coords[1]  # tuile à droite
        # overlap region entre tuile 0 et tuile 1
        ov_x = x0 + stride
        ov_rect = patches.Rectangle(
            (ov_x, y0), overlap, patch_size,
            linewidth=1.5, edgecolor="yellow", facecolor="yellow", alpha=0.35
        )
        ax.add_patch(ov_rect)

    # Annotations dimensions
    y_anno = H * 0.02
    ax.annotate(
        f"patch {patch_size}×{patch_size}",
        xy=(coords[1][1] + patch_size / 2, coords[0][0] + patch_size / 2),
        fontsize=8, color="white", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6)
    )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(
        f"Découpage en tuiles — {W}×{H} → {n_tiles} tuiles ({n_y} lignes × {n_x} cols)\n"
        f"patch={patch_size}×{patch_size} | overlap={overlap}px | stride={stride}px",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("x (pixels)", fontsize=9)
    ax.set_ylabel("y (pixels)", fontsize=9)

    # Légende
    handles = [
        patches.Patch(facecolor="steelblue", edgecolor="white", alpha=0.5, label="Tuiles"),
        patches.Patch(facecolor="yellow",    edgecolor="black",  alpha=0.5, label=f"Zone d'overlap ({overlap}px)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    out1 = os.path.join(args.out_dir, "tile_grid_overview.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out1}")

    # =========================================================================
    # Figure 2 — Zoom sur 3×2 tuiles avec overlap clairement visible
    # =========================================================================
    # Prendre les 6 premières tuiles (3 cols × 2 lignes)
    # Crop sur la zone correspondante dans l'image
    n_show_x = min(3, n_x)
    n_show_y = min(2, n_y)
    coords_show = [c for c in coords if c[1] // stride < n_show_x and c[0] // stride < n_show_y]

    x_min = min(c[1] for c in coords_show)
    x_max = max(c[1] for c in coords_show) + patch_size
    y_min = min(c[0] for c in coords_show)
    y_max = max(c[0] for c in coords_show) + patch_size
    x_max = min(x_max, W)
    y_max = min(y_max, H)

    crop_zoom = frame_img[y_min:y_max, x_min:x_max]

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.imshow(crop_zoom, alpha=0.80)

    tile_colors_named = [
        ("#3498db", "Tuile A"),
        ("#e74c3c", "Tuile B"),
        ("#2ecc71", "Tuile C"),
        ("#9b59b6", "Tuile D"),
        ("#f39c12", "Tuile E"),
        ("#1abc9c", "Tuile F"),
    ]

    for i, (y0, x0) in enumerate(coords_show):
        col, lbl = tile_colors_named[i % len(tile_colors_named)]
        # Convertir en coord locales du crop
        lx, ly = x0 - x_min, y0 - y_min

        # Fond coloré semi-transparent
        rect_fill = patches.Rectangle(
            (lx, ly), patch_size, patch_size,
            linewidth=0, facecolor=col, alpha=0.15
        )
        ax2.add_patch(rect_fill)

        # Contour épais
        rect_border = patches.Rectangle(
            (lx, ly), patch_size, patch_size,
            linewidth=2.0, edgecolor=col, facecolor="none", alpha=0.9
        )
        ax2.add_patch(rect_border)

        # Label centré
        ax2.text(
            lx + patch_size / 2, ly + patch_size / 2, lbl,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.3", fc=col, alpha=0.8)
        )

    # Dessiner les zones d'overlap horizontales entre tuiles adjacentes
    for i, (y0a, x0a) in enumerate(coords_show):
        for j, (y0b, x0b) in enumerate(coords_show):
            if i >= j:
                continue
            # Overlap horizontal (même ligne, tuile à droite)
            if y0a == y0b and x0b == x0a + stride:
                lx = x0a + stride - x_min
                ly = y0a - y_min
                ov_h = patches.Rectangle(
                    (lx, ly), overlap, patch_size,
                    linewidth=1.5, edgecolor="#f1c40f", facecolor="#f1c40f", alpha=0.45
                )
                ax2.add_patch(ov_h)
                ax2.text(lx + overlap / 2, ly + patch_size / 2,
                         f"{overlap}px\noverlay", ha="center", va="center",
                         fontsize=7, color="black",
                         bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.9))

            # Overlap vertical (même colonne, tuile en dessous)
            if x0a == x0b and y0b == y0a + stride:
                lx = x0a - x_min
                ly = y0a + stride - y_min
                ov_v = patches.Rectangle(
                    (lx, ly), patch_size, overlap,
                    linewidth=1.5, edgecolor="#f1c40f", facecolor="#f1c40f", alpha=0.45
                )
                ax2.add_patch(ov_v)
                ax2.text(lx + patch_size / 2, ly + overlap / 2,
                         f"{overlap}px overlay", ha="center", va="center",
                         fontsize=7, color="black",
                         bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.9))

    ax2.set_xlim(0, x_max - x_min)
    ax2.set_ylim(y_max - y_min, 0)

    # Flèche annotation stride
    ax2.annotate(
        "", xy=(stride, -18), xytext=(0, -18),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5)
    )
    ax2.text(stride / 2, -28, f"stride = {stride}px", ha="center", color="white", fontsize=8)

    ax2.set_title(
        f"Zoom sur {n_show_y}×{n_show_x} tuiles — overlap={overlap}px (en jaune) entre tuiles adjacentes\n"
        f"Chaque pixel dans l'overlap est prédit par 2 tuiles → on fait la moyenne",
        fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("x (pixels)", fontsize=9)
    ax2.set_ylabel("y (pixels)", fontsize=9)

    handles2 = [
        patches.Patch(facecolor="steelblue", edgecolor="white", alpha=0.6, label="Tuile (256×256)"),
        patches.Patch(facecolor="yellow",    edgecolor="black",  alpha=0.6, label=f"Zone d'overlap ({overlap}px) — moyennée"),
    ]
    ax2.legend(handles=handles2, loc="lower right", fontsize=9)

    out2 = os.path.join(args.out_dir, "tile_overlap_zoom.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out2}")

    # =========================================================================
    # Figure 3 — Schéma 1D pédagogique (strip horizontal)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(13, 3.5))
    ax3.set_xlim(0, W)
    ax3.set_ylim(-60, 320)
    ax3.axis("off")
    ax3.set_facecolor("#1a1a2e")
    fig3.patch.set_facecolor("#1a1a2e")

    # Bande image
    image_bar = patches.Rectangle((0, 100), W, 100, facecolor="#2c3e50", edgecolor="white", linewidth=1)
    ax3.add_patch(image_bar)
    ax3.text(W / 2, 150, f"Image 1920×1080", ha="center", va="center",
             color="white", fontsize=12, fontweight="bold")

    # Tuiles avec overlaps
    tile_cols = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c",
                 "#e67e22", "#27ae60", "#8e44ad"]
    x_positions = []
    x = 0
    while True:
        x0 = min(x, W - patch_size)
        x_positions.append(x0)
        if x0 + patch_size >= W:
            break
        x += stride

    for i, x0 in enumerate(x_positions):
        col = tile_cols[i % len(tile_cols)]
        # Tuile sous l'image
        rect = patches.FancyBboxPatch(
            (x0, -50), patch_size, 80,
            boxstyle="round,pad=3",
            facecolor=col, edgecolor="white", linewidth=1.2, alpha=0.85
        )
        ax3.add_patch(rect)
        if i < len(x_positions) - 1:
            ax3.text(x0 + patch_size / 2, -10, f"T{i+1}", ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")

    # Zones d'overlap entre tuiles
    for i, x0 in enumerate(x_positions[:-1]):
        ov_x = x0 + stride
        ov_rect = patches.Rectangle(
            (ov_x, -50), overlap, 80,
            facecolor="#f1c40f", edgecolor="black", linewidth=1.0, alpha=0.7
        )
        ax3.add_patch(ov_rect)

    # Annotations
    if len(x_positions) >= 2:
        x0a, x0b = x_positions[0], x_positions[1]
        ax3.annotate("", xy=(x0b, 215), xytext=(x0a, 215),
                     arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
        ax3.text((x0a + x0b) / 2, 230, f"stride = {stride}px", ha="center", color="white", fontsize=8)

        ax3.annotate("", xy=(x0a + stride + overlap, 260), xytext=(x0a + stride, 260),
                     arrowprops=dict(arrowstyle="<->", color="#f1c40f", lw=2))
        ax3.text(x0a + stride + overlap / 2, 280, f"overlap\n{overlap}px",
                 ha="center", color="#f1c40f", fontsize=8, fontweight="bold")

    ax3.text(W / 2, 310,
             f"{len(x_positions)} tuiles horizontales | patch={patch_size}px | stride={stride}px | overlap={overlap}px",
             ha="center", va="center", color="white", fontsize=9)

    handles3 = [
        patches.Patch(facecolor="#3498db", edgecolor="white", alpha=0.8, label=f"Tuile {patch_size}px"),
        patches.Patch(facecolor="#f1c40f", edgecolor="black", alpha=0.8, label=f"Overlap {overlap}px (moyenné)"),
    ]
    ax3.legend(handles=handles3, loc="upper left", fontsize=9,
               facecolor="#2c3e50", edgecolor="white", labelcolor="white")

    ax3.set_title(f"Schéma 1D du découpage — {len(x_positions)} tuiles en horizontal sur {W}px",
                  fontsize=11, fontweight="bold", color="white")

    out3 = os.path.join(args.out_dir, "tile_schema_1d.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close()
    print(f"Sauvegardé : {out3}")

    print(f"\nRésumé final :")
    print(f"  Résolution : {W}×{H}")
    print(f"  patch_size : {patch_size}×{patch_size}")
    print(f"  overlap    : {overlap}px")
    print(f"  stride     : {stride}px")
    print(f"  Tuiles     : {n_tiles} ({n_y}×{n_x})")


if __name__ == "__main__":
    main()
