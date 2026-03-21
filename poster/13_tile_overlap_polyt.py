"""
poster/13_tile_overlap_polyt.py — tile_overlap_zoom sur polyt.jpg (crop 1920x1080)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

RESULTS = os.path.join(os.path.dirname(__file__), "results")
IMG_PATH = os.path.join(RESULTS, "polyt.jpg")
OUT_PATH = os.path.join(RESULTS, "tile_overlap_zoom_polyt.png")

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP
W, H       = 1920, 1080


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


# --- Charger et crop 1920x1080 centré ---
img = Image.open(IMG_PATH).convert("RGB")
iw, ih = img.size
# Crop centré pour respecter le ratio 1920x1080
target_ratio = W / H
img_ratio = iw / ih

if img_ratio > target_ratio:
    # Image trop large → crop en largeur
    new_w = int(ih * target_ratio)
    left  = (iw - new_w) // 2
    img   = img.crop((left, 0, left + new_w, ih))
else:
    # Image trop haute → crop en hauteur
    new_h = int(iw / target_ratio)
    top   = (ih - new_h) // 2
    img   = img.crop((0, top, iw, top + new_h))

img = img.resize((W, H), Image.LANCZOS)
frame_img = np.array(img)

# --- Zoom sur 3x2 tuiles ---
coords = get_patch_coords(H, W, PATCH_SIZE, OVERLAP)
n_x = len(set(c[1] for c in coords))
n_y = len(set(c[0] for c in coords))

n_show_x = min(3, n_x)
n_show_y = min(2, n_y)
coords_show = [c for c in coords
               if c[1] // STRIDE < n_show_x and c[0] // STRIDE < n_show_y]

x_min = min(c[1] for c in coords_show)
x_max = min(max(c[1] for c in coords_show) + PATCH_SIZE, W)
y_min = min(c[0] for c in coords_show)
y_max = min(max(c[0] for c in coords_show) + PATCH_SIZE, H)

crop_zoom = frame_img[y_min:y_max, x_min:x_max]

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(crop_zoom, alpha=0.85)

tile_colors = [
    ("#3498db", "Tile A"), ("#e74c3c", "Tile B"), ("#2ecc71", "Tile C"),
    ("#9b59b6", "Tile D"), ("#f39c12", "Tile E"), ("#1abc9c", "Tile F"),
]

for i, (y0, x0) in enumerate(coords_show):
    col, lbl = tile_colors[i % len(tile_colors)]
    lx, ly = x0 - x_min, y0 - y_min

    ax.add_patch(patches.Rectangle((lx, ly), PATCH_SIZE, PATCH_SIZE,
                                    linewidth=0, facecolor=col, alpha=0.15))
    ax.add_patch(patches.Rectangle((lx, ly), PATCH_SIZE, PATCH_SIZE,
                                    linewidth=2.0, edgecolor=col, facecolor="none", alpha=0.9))
    ax.text(lx + PATCH_SIZE / 2, ly + PATCH_SIZE / 2, lbl,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.3", fc=col, alpha=0.8))

# Overlap zones
for i, (y0a, x0a) in enumerate(coords_show):
    for j, (y0b, x0b) in enumerate(coords_show):
        if i >= j:
            continue
        if y0a == y0b and x0b == x0a + STRIDE:
            lx = x0a + STRIDE - x_min
            ly = y0a - y_min
            ax.add_patch(patches.Rectangle((lx, ly), OVERLAP, PATCH_SIZE,
                                            linewidth=1.5, edgecolor="#f1c40f",
                                            facecolor="#f1c40f", alpha=0.45))
            ax.text(lx + OVERLAP / 2, ly + PATCH_SIZE / 2,
                    f"{OVERLAP}px\noverlap", ha="center", va="center",
                    fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.9))
        if x0a == x0b and y0b == y0a + STRIDE:
            lx = x0a - x_min
            ly = y0a + STRIDE - y_min
            ax.add_patch(patches.Rectangle((lx, ly), PATCH_SIZE, OVERLAP,
                                            linewidth=1.5, edgecolor="#f1c40f",
                                            facecolor="#f1c40f", alpha=0.45))
            ax.text(lx + PATCH_SIZE / 2, ly + OVERLAP / 2,
                    f"{OVERLAP}px overlap", ha="center", va="center",
                    fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.9))

ax.set_xlim(0, x_max - x_min)
ax.set_ylim(y_max - y_min, 0)
ax.set_title(
    f"Tiling with overlap — patch={PATCH_SIZE}×{PATCH_SIZE}px | overlap={OVERLAP}px (yellow) | stride={STRIDE}px\n"
    f"Pixels in overlap regions are predicted by 2 tiles and averaged",
    fontsize=11, fontweight="bold"
)
ax.set_xlabel("x (pixels)", fontsize=9)
ax.set_ylabel("y (pixels)", fontsize=9)

handles = [
    patches.Patch(facecolor="steelblue", edgecolor="white", alpha=0.6, label=f"Tile (256×256px)"),
    patches.Patch(facecolor="yellow",    edgecolor="black",  alpha=0.6, label=f"Overlap ({OVERLAP}px) — averaged"),
]
ax.legend(handles=handles, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
