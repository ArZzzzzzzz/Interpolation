"""
poster/02_tile_demo.py — Démonstration de l'inférence par tuiles (tiled inference)

Problème : le modèle est entraîné sur des patches 256×256.
Solution : pour une vidéo 1920×1080, on découpe en tuiles avec overlap,
           on interpole chaque tuile, et on moyenne les zones superposées.

Génère : results/tile_demo.png
         results/tile_comparison.png  (zoom sur une zone avec/sans tiling)

Usage:
    cd poster
    python 02_tile_demo.py --video "../19201080_before.mp4"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2


# ---------------------------------------------------------------------------
# Tiling utilities (repris de double_fps.py)
# ---------------------------------------------------------------------------

def img_to_tensor(img, device):
    t = TF.to_tensor(img) * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def tensor_to_np(t):
    t = t.squeeze(0).cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0).permute(1, 2, 0).numpy()


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


@torch.no_grad()
def tiled_interpolate(model, I0_pil, I1_pil, patch_size, overlap, device):
    W, H = I0_pil.size
    arr0 = np.array(I0_pil).astype(np.float32) / 255.0
    arr1 = np.array(I1_pil).astype(np.float32) / 255.0

    accum   = np.zeros((H, W, 3), dtype=np.float32)
    weights = np.zeros((H, W, 1), dtype=np.float32)

    coords = get_patch_coords(H, W, patch_size, overlap)
    n_tiles = len(coords)

    for (y0, x0) in coords:
        p0 = arr0[y0:y0+patch_size, x0:x0+patch_size]
        p1 = arr1[y0:y0+patch_size, x0:x0+patch_size]

        t0 = img_to_tensor(Image.fromarray((p0 * 255).astype(np.uint8)), device)
        t1 = img_to_tensor(Image.fromarray((p1 * 255).astype(np.uint8)), device)

        pred = model(t0, t1).clamp(-1, 1)
        pred_np = tensor_to_np(pred)

        accum[y0:y0+patch_size,   x0:x0+patch_size]   += pred_np
        weights[y0:y0+patch_size, x0:x0+patch_size]   += 1.0

    result = (accum / weights.clip(min=1e-6)).clip(0, 1)
    return Image.fromarray((result * 255).astype(np.uint8)), coords, n_tiles


@torch.no_grad()
def naive_interpolate(model, I0_pil, I1_pil, size, orig_size, device):
    """Redimensionne à size×size, interpole, remet à orig_size."""
    I0r = I0_pil.resize((size, size), Image.LANCZOS)
    I1r = I1_pil.resize((size, size), Image.LANCZOS)
    t0  = img_to_tensor(I0r, device)
    t1  = img_to_tensor(I1r, device)
    pred = model(t0, t1).clamp(-1, 1)
    pred_np = tensor_to_np(pred)
    pil = Image.fromarray((pred_np * 255).astype(np.uint8))
    return pil.resize(orig_size, Image.LANCZOS)


def load_pixel_v2(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = UNetPixelV2(
        base_ch      = args_.get("base_ch", 64),
        use_residual = not args_.get("no_residual", False),
        use_refine   = not args_.get("no_refine", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      type=str, default=os.path.join(ROOT, "19201080_before.mp4"))
    p.add_argument("--pixel_ckpt", type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--overlap",    type=int, default=32)
    p.add_argument("--frame_idx",  type=int, default=10, help="Indice de la paire de frames à utiliser")
    p.add_argument("--out_dir",    type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Charger 2 frames depuis la vidéo
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir {args.video}")
        print("Utilisation de frames synthétiques pour la démo...")
        # Créer des frames synthétiques
        W, H = 1920, 1080
        arr0 = np.random.randint(80, 180, (H, W, 3), dtype=np.uint8)
        arr1 = np.clip(arr0.astype(int) + np.random.randint(-20, 20, (H, W, 3)), 0, 255).astype(np.uint8)
        I0 = Image.fromarray(arr0)
        I1 = Image.fromarray(arr1)
    else:
        for _ in range(args.frame_idx):
            cap.read()
        ret0, f0 = cap.read()
        ret1, f1 = cap.read()
        cap.release()
        I0 = Image.fromarray(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
        I1 = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))

    W, H = I0.size
    print(f"Résolution : {W}×{H}")

    # Charger modèle
    model = load_pixel_v2(args.pixel_ckpt, device)

    # Tiled inference
    print(f"Inférence tiled ({args.patch_size}×{args.patch_size}, overlap={args.overlap})...")
    Imid_tiled, coords, n_tiles = tiled_interpolate(model, I0, I1, args.patch_size, args.overlap, device)
    print(f"  {n_tiles} tuiles traitées")

    # Naive inference (resize)
    print("Inférence naïve (resize 256, interpolation, resize back)...")
    Imid_naive = naive_interpolate(model, I0, I1, 256, (W, H), device)

    # Average baseline
    arr0_f = np.array(I0).astype(np.float32)
    arr1_f = np.array(I1).astype(np.float32)
    Imid_avg = Image.fromarray(((arr0_f + arr1_f) / 2).clip(0, 255).astype(np.uint8))

    # ---- Figure 1 : grille de tuiles sur I0 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Inférence par tuiles — {W}×{H} → {n_tiles} tuiles {args.patch_size}×{args.patch_size} (overlap={args.overlap}px)",
                 fontsize=12, fontweight="bold")

    # I0 avec grille de tuiles
    ax = axes[0]
    ax.imshow(I0)
    stride = args.patch_size - args.overlap
    n_x = len(set(c[1] for c in coords))
    n_y = len(set(c[0] for c in coords))
    for (y0, x0) in coords:
        rect = patches.Rectangle((x0, y0), args.patch_size, args.patch_size,
                                  linewidth=0.4, edgecolor='yellow', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    ax.set_title(f"I₀ avec grille de tuiles ({n_y}×{n_x} = {n_tiles} tuiles)", fontsize=10)
    ax.axis("off")

    # Résultat tiled
    axes[1].imshow(Imid_tiled)
    axes[1].set_title(f"Imid — inférence tiled (qualité HD)", fontsize=10)
    axes[1].axis("off")

    # Résultat naïf
    axes[2].imshow(Imid_naive)
    axes[2].set_title("Imid — inférence naïve\n(resize 256→interp→resize back)", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "tile_demo.png")
    plt.savefig(out1, dpi=100, bbox_inches="tight")
    print(f"Sauvegardé : {out1}")
    plt.close()

    # ---- Figure 2 : zoom comparatif sur une zone riche en détails ----
    # Prendre un crop 256×256 au milieu
    cx, cy = W // 2, H // 2
    margin = 192
    crop_box = (cx - margin, cy - margin, cx + margin, cy + margin)

    I0_crop      = I0.crop(crop_box)
    I1_crop      = I1.crop(crop_box)
    tiled_crop   = Imid_tiled.crop(crop_box)
    naive_crop   = Imid_naive.crop(crop_box)
    avg_crop     = Imid_avg.crop(crop_box)

    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
    fig2.suptitle(f"Zoom central ({margin*2}×{margin*2}px) — comparaison des approches", fontsize=12, fontweight="bold")

    for ax, img, title in zip(axes2,
        [I0_crop, I1_crop, avg_crop, naive_crop, tiled_crop],
        ["I₀", "I₁", "Moyenne (baseline)", "Naïf (resize)", "Tiled (notre approche)"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "tile_comparison.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Sauvegardé : {out2}")
    plt.close()

    # ---- Figure 3 : carte de différence tiled vs naïf ----
    arr_tiled = np.array(Imid_tiled).astype(float)
    arr_naive = np.array(Imid_naive).astype(float)
    diff = np.abs(arr_tiled - arr_naive).mean(axis=2)   # [H, W]

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))
    fig3.suptitle("Différence entre inférence tiled et naïve (erreur amplifiée ×5)", fontsize=11)

    axes3[0].imshow(Imid_tiled)
    axes3[0].set_title("Inférence tiled", fontsize=10)
    axes3[0].axis("off")

    im = axes3[1].imshow((diff * 5).clip(0, 255), cmap="hot", vmin=0, vmax=60)
    axes3[1].set_title("Carte de différence |tiled − naïf|", fontsize=10)
    axes3[1].axis("off")
    plt.colorbar(im, ax=axes3[1], fraction=0.03, pad=0.02)

    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "tile_diff.png")
    plt.savefig(out3, dpi=100, bbox_inches="tight")
    print(f"Sauvegardé : {out3}")
    plt.close()

    print(f"\nRésumé : {n_tiles} tuiles, résolution {W}×{H}")
    print("Images générées dans results/")


if __name__ == "__main__":
    main()
