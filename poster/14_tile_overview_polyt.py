"""
poster/14_tile_overview_polyt.py — tile_grid_overview sur polyt.jpg (crop 1920x1080)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

RESULTS  = os.path.join(os.path.dirname(__file__), "results")
IMG_PATH = os.path.join(RESULTS, "polyt.jpg")
OUT_PATH = os.path.join(RESULTS, "tile_grid_overview_polyt.png")

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


# Charger et crop 1920x1080 centré
img = Image.open(IMG_PATH).convert("RGB")
iw, ih = img.size
target_ratio = W / H
img_ratio    = iw / ih
if img_ratio > target_ratio:
    new_w = int(ih * target_ratio)
    left  = (iw - new_w) // 2
    img   = img.crop((left, 0, left + new_w, ih))
else:
    new_h = int(iw / target_ratio)
    top   = (ih - new_h) // 2
    img   = img.crop((0, top, iw, top + new_h))
img = img.resize((W, H), Image.LANCZOS)
frame_img = np.array(img)

coords = get_patch_coords(H, W, PATCH_SIZE, OVERLAP)
n_x = len(set(c[1] for c in coords))
n_y = len(set(c[0] for c in coords))

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(frame_img, alpha=0.75)

colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
for i, (y0, x0) in enumerate(coords):
    col = colors[i % len(colors)]
    ax.add_patch(patches.Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE,
                                    linewidth=0.6, edgecolor=col, facecolor=col, alpha=0.08))
    ax.add_patch(patches.Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE,
                                    linewidth=0.6, edgecolor="white", facecolor="none", alpha=0.6))

# Overlap highlight entre les 2 premières tuiles
if len(coords) >= 2:
    y0, x0 = coords[0]
    ov_x = x0 + STRIDE
    ax.add_patch(patches.Rectangle((ov_x, y0), OVERLAP, PATCH_SIZE,
                                    linewidth=1.5, edgecolor="yellow",
                                    facecolor="yellow", alpha=0.35))

ax.annotate(
    f"patch {PATCH_SIZE}×{PATCH_SIZE}",
    xy=(coords[1][1] + PATCH_SIZE / 2, coords[0][0] + PATCH_SIZE / 2),
    fontsize=8, color="white", ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6)
)

ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_title(
    f"Tiling — {W}×{H} → {len(coords)} tiles ({n_y} rows × {n_x} cols)\n"
    f"patch={PATCH_SIZE}×{PATCH_SIZE} | overlap={OVERLAP}px | stride={STRIDE}px",
    fontsize=11, fontweight="bold"
)
ax.set_xlabel("x (pixels)", fontsize=9)
ax.set_ylabel("y (pixels)", fontsize=9)

handles = [
    patches.Patch(facecolor="steelblue", edgecolor="white", alpha=0.5, label="Tiles"),
    patches.Patch(facecolor="yellow",    edgecolor="black",  alpha=0.5, label=f"Overlap ({OVERLAP}px)"),
]
ax.legend(handles=handles, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
