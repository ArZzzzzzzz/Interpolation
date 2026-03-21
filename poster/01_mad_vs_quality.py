"""
poster/01_mad_vs_quality.py — MAD inter-frames vs qualité de prédiction

Montre qu'au-delà d'un certain mouvement (MAD élevé entre I0 et I1),
le modèle pixel prédit une "moyenne floue" et le PSNR chute.

Génère : results/mad_vs_quality.png

Usage:
    cd poster
    python 01_mad_vs_quality.py --data_root "D:/projet mumltimodal/vimeo_triplet"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2
from metrics import compute_psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_np255(t):
    """[B,3,H,W] in [-1,1] → numpy [B,H,W,3] in [0,255]."""
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0 * 255.0
    return t.permute(0, 2, 3, 1).numpy()


def compute_mad(I0_np, I1_np):
    """MAD en [0,255] entre I0 et I1 par image du batch."""
    return np.mean(np.abs(I0_np - I1_np), axis=(1, 2, 3))   # [B]


def compute_farneback_flow(I0_np255, I1_np255):
    """Retourne flow [H,W,2] en pixels."""
    g0 = cv2.cvtColor(I0_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(I1_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_tensor(img_t, flow_np, t=0.5):
    """Warp img_t par t*flow_np. img_t: [1,3,H,W] in [-1,1], flow_np: [H,W,2]."""
    H, W = img_t.shape[-2:]
    flow_t = torch.from_numpy(flow_np).float()
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    flow_x = t * flow_t[..., 0] / (W / 2)
    flow_y = t * flow_t[..., 1] / (H / 2)
    grid = torch.stack([gx + flow_x, gy + flow_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def optical_flow_interp(I0_t, I1_t, I0_np255, I1_np255):
    """Interpolation par flot optique Farneback. Retourne [1,3,H,W] in [-1,1]."""
    flow = compute_farneback_flow(I0_np255, I1_np255)
    w0 = warp_tensor(I0_t.unsqueeze(0), flow, t=0.5)
    w1 = warp_tensor(I1_t.unsqueeze(0), flow, t=-0.5)
    return (0.5 * w0 + 0.5 * w1).clamp(-1, 1)


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
    p.add_argument("--data_root", type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--pixel_ckpt", type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--n_batches", type=int, default=200, help="Nombre de batches à évaluer")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "results/mad_vs_quality.png"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Device: {device}")

    # Data
    _, val_loader, _ = get_dataloaders(args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)

    # Models
    model = load_pixel_v2(args.pixel_ckpt, device)
    print(f"UNetPixelV2 chargé depuis {args.pixel_ckpt}")

    mad_list, psnr_pixel_list, psnr_avg_list, psnr_of_list = [], [], [], []

    print(f"Évaluation sur {args.n_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_batches:
                break

            I0   = batch["I0"].to(device)
            Imid = batch["Imid"].to(device)
            I1   = batch["I1"].to(device)

            # MAD en [0,255]
            I0_np   = tensor_to_np255(I0)
            I1_np   = tensor_to_np255(I1)
            Imid_np = tensor_to_np255(Imid)
            mad     = compute_mad(I0_np, I1_np)   # [B]

            # Prédictions deep learning
            pred_pixel = model(I0, I1).clamp(-1, 1)
            pred_avg   = 0.5 * (I0 + I1)

            # PSNR image par image
            for b in range(I0.shape[0]):
                p  = pred_pixel[b:b+1]
                gt = Imid[b:b+1]
                av = pred_avg[b:b+1]
                psnr_pixel_list.append(compute_psnr(p, gt))
                psnr_avg_list.append(compute_psnr(av, gt))
                mad_list.append(float(mad[b]))

                # Optical flow (CPU, image par image)
                pred_of = optical_flow_interp(
                    I0[b].cpu(), I1[b].cpu(), I0_np[b], I1_np[b]
                )
                psnr_of_list.append(compute_psnr(pred_of, gt.cpu()))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{args.n_batches} batches")

    mad_arr   = np.array(mad_list)
    psnr_px   = np.array(psnr_pixel_list)
    psnr_av   = np.array(psnr_avg_list)
    psnr_of   = np.array(psnr_of_list)

    print(f"\nTotal échantillons : {len(mad_arr)}")
    print(f"MAD   : min={mad_arr.min():.1f} | médiane={np.median(mad_arr):.1f} | max={mad_arr.max():.1f}")
    print(f"PSNR PixelV2 moyen : {psnr_px.mean():.2f} dB")
    print(f"PSNR Average moyen : {psnr_av.mean():.2f} dB")
    print(f"PSNR Optical Flow  : {psnr_of.mean():.2f} dB")

    # ---- Binning ----
    bins = np.linspace(0, min(mad_arr.max(), 60), 16)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(mad_arr, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    psnr_px_binned  = [psnr_px[bin_idx == k].mean() if (bin_idx == k).sum() > 3 else np.nan for k in range(len(bins)-1)]
    psnr_av_binned  = [psnr_av[bin_idx == k].mean() if (bin_idx == k).sum() > 3 else np.nan for k in range(len(bins)-1)]
    psnr_of_binned  = [psnr_of[bin_idx == k].mean() if (bin_idx == k).sum() > 3 else np.nan for k in range(len(bins)-1)]
    counts          = [(bin_idx == k).sum() for k in range(len(bins)-1)]

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Influence du mouvement inter-frames (MAD) sur la qualité de prédiction",
                 fontsize=13, fontweight="bold")

    # Scatter (sous-échantillonné pour la lisibilité)
    ax = axes[0]
    idx_show = np.random.choice(len(mad_arr), min(3000, len(mad_arr)), replace=False)
    ax.scatter(mad_arr[idx_show], psnr_px[idx_show], s=4, alpha=0.3, color="steelblue",  label="UNetPixelV2")
    ax.scatter(mad_arr[idx_show], psnr_av[idx_show], s=4, alpha=0.2, color="coral",      label="Moyenne (baseline)")
    ax.scatter(mad_arr[idx_show], psnr_of[idx_show], s=4, alpha=0.2, color="darkorange", label="Optical Flow (Farneback)")
    ax.set_xlabel("MAD (I0 <-> I1) [0-255]", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("Scatter : PSNR vs MAD", fontsize=11)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=20)

    # Binned curves
    ax2 = axes[1]
    valid = [i for i in range(len(bin_centers)) if not np.isnan(psnr_px_binned[i])]
    xv   = bin_centers[valid]
    ax2.plot(xv, [psnr_px_binned[i] for i in valid], "o-", color="steelblue",  linewidth=2, label="UNetPixelV2")
    ax2.plot(xv, [psnr_av_binned[i] for i in valid], "s--", color="coral",     linewidth=2, label="Moyenne (I0+I1)/2")
    ax2.plot(xv, [psnr_of_binned[i] for i in valid], "^--", color="darkorange", linewidth=2, label="Optical Flow (Farneback)")
    ax2.fill_between(xv, [psnr_av_binned[i] for i in valid], [psnr_px_binned[i] for i in valid],
                     alpha=0.15, color="steelblue", label="Gain PixelV2 vs Moyenne")

    # Seuil : à partir de quel MAD l'écart rétrécit ?
    ax2.axvline(15, color="red", linestyle="--", linewidth=1.2, label="Seuil approx. 15 (mouvement fort)")
    ax2.set_xlabel("MAD (I0 <-> I1) [0-255]", fontsize=11)
    ax2.set_ylabel("PSNR moyen (dB)", fontsize=11)
    ax2.set_title("PSNR moyen par intervalle de MAD", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=20)

    # Histogramme en fond (densité des données)
    ax2b = ax2.twinx()
    ax2b.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.8, alpha=0.12, color="gray")
    ax2b.set_ylabel("Nb. d'échantillons", fontsize=8, color="gray")
    ax2b.tick_params(axis='y', labelcolor='gray', labelsize=7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {args.out}")


if __name__ == "__main__":
    main()
