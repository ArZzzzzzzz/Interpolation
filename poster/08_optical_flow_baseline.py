"""
poster/08_optical_flow_baseline.py — Baseline Optical Flow (style Premiere Pro)

Pipeline :
    1. flow = Farneback(I0_gray, I1_gray)        # vecteurs de déplacement (dx, dy)
    2. I0_warped = warp(I0,  0.5 * flow)          # I0 projeté vers t=0.5
    3. I1_warped = warp(I1, -0.5 * flow)          # I1 projeté vers t=0.5
    4. Imid_pred = 0.5 * I0_warped + 0.5 * I1_warped

Compare avec toutes les autres méthodes :
  - Moyenne simple (I0+I1)/2
  - Optical Flow (Farneback) ← cette baseline
  - UNetPixelV2
  - UNetPixelV2 + GAN

Génère :
  results/optical_flow_comparison_XXX.png
  results/full_table.png   (tableau complet avec optical flow)

Usage:
    cd poster
    python 08_optical_flow_baseline.py --data_root "D:/projet mumltimodal/vimeo_triplet"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2
from metrics import compute_psnr, compute_ssim, LPIPSMetric, MetricAccumulator


# ---------------------------------------------------------------------------
# Optical flow warp (CPU via OpenCV + PyTorch grid_sample)
# ---------------------------------------------------------------------------

def tensor_to_np_uint8(t):
    """[3,H,W] dans [-1,1] → np uint8 [H,W,3]."""
    arr = ((t.cpu().clamp(-1, 1) + 1.0) / 2.0 * 255).byte().permute(1, 2, 0).numpy()
    return arr.copy()


def compute_farneback_flow(I0_np, I1_np):
    """
    Calcule le flot optique Farneback de I0 vers I1.
    Retourne flow [H, W, 2] en pixels (dx, dy).
    """
    g0 = cv2.cvtColor(I0_np, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(I1_np, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1,
        flow=None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    return flow   # [H, W, 2] : flow[...,0]=dx, flow[...,1]=dy


def warp_tensor(img_t, flow_np, t=0.5):
    """
    Warp img_t selon t * flow_np via grid_sample (bilinéaire).

    img_t  : [3, H, W] dans [-1, 1]
    flow_np: [H, W, 2] en pixels (dx, dy)
    t      : fraction temporelle (0.5 → mi-chemin)
    """
    _, H, W = img_t.shape
    device = img_t.device

    # Grille de base (coordonnées normalisées [-1,1])
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    # Convertir flow pixels → coordonnées normalisées
    flow_t = torch.from_numpy(flow_np).to(device).float()     # [H, W, 2]
    flow_x = flow_t[..., 0] * t * (2.0 / W)
    flow_y = flow_t[..., 1] * t * (2.0 / H)

    # Grille déplacée
    grid_xw = (grid_x + flow_x).clamp(-1, 1)
    grid_yw = (grid_y + flow_y).clamp(-1, 1)
    grid = torch.stack([grid_xw, grid_yw], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    img_b = img_t.unsqueeze(0)  # [1, 3, H, W]
    warped = F.grid_sample(img_b, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze(0)    # [3, H, W]


@torch.no_grad()
def optical_flow_interp(I0_t, I1_t):
    """
    I0_t, I1_t : [3, H, W] dans [-1, 1] (tenseurs CPU ou GPU)
    Retourne Imid_pred [3, H, W] dans [-1, 1]
    """
    I0_np = tensor_to_np_uint8(I0_t)
    I1_np = tensor_to_np_uint8(I1_t)

    flow_fwd = compute_farneback_flow(I0_np, I1_np)   # I0 -> I1

    I0_warped = warp_tensor(I0_t, flow_fwd,  t= 0.5)   # I0 vers t=0.5
    I1_warped = warp_tensor(I1_t, flow_fwd,  t=-0.5)   # I1 vers t=0.5 (flow inversé)

    return (0.5 * I0_warped + 0.5 * I1_warped).clamp(-1, 1)


def batch_optical_flow(I0_batch, I1_batch):
    """Applique optical_flow_interp sur chaque image du batch."""
    preds = []
    for b in range(I0_batch.shape[0]):
        pred = optical_flow_interp(I0_batch[b], I1_batch[b])
        preds.append(pred)
    return torch.stack(preds, dim=0)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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


def load_gan(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNetPixelV2(base_ch=64).to(device)
    model.load_state_dict(ckpt["G"])
    model.eval()
    return model


def to_np(t):
    if t.dim() == 4:
        t = t.squeeze(0)
    return ((t.cpu().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--pixel_ckpt", type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--gan_ckpt",   type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2_gan/best.pth"))
    p.add_argument("--n_eval",     type=int, default=100)
    p.add_argument("--n_samples",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out_dir",    type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    _, val_loader, _ = get_dataloaders(
        args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)

    print("\nChargement des modèles...")
    pixel_model = load_pixel_v2(args.pixel_ckpt, device)
    gan_model   = load_gan(args.gan_ckpt, device) if os.path.isfile(args.gan_ckpt) else None
    print("  OK")

    acc_avg  = MetricAccumulator()
    acc_of   = MetricAccumulator()
    acc_px   = MetricAccumulator()
    acc_gan  = MetricAccumulator()
    lpips_fn = LPIPSMetric().to(device)

    samples = []

    print(f"\nEvaluation sur {args.n_eval} batches...")
    for i, batch in enumerate(val_loader):
        if i >= args.n_eval:
            break

        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        pred_avg = (0.5 * (I0 + I1)).clamp(-1, 1)

        # Optical flow (CPU, image par image)
        pred_of = batch_optical_flow(I0, I1).to(device)

        with torch.no_grad():
            pred_px  = pixel_model(I0, I1).clamp(-1, 1)
            pred_gan = gan_model(I0, I1).clamp(-1, 1) if gan_model else None

        for key, pred, acc in [
            ("avg",  pred_avg, acc_avg),
            ("of",   pred_of,  acc_of),
            ("px",   pred_px,  acc_px),
        ]:
            acc.update(
                psnr  = compute_psnr(pred, Imid),
                ssim  = compute_ssim(pred.cpu(), Imid.cpu()),
                lpips = lpips_fn(pred, Imid),
            )
        if pred_gan is not None:
            acc_gan.update(
                psnr  = compute_psnr(pred_gan, Imid),
                ssim  = compute_ssim(pred_gan.cpu(), Imid.cpu()),
                lpips = lpips_fn(pred_gan, Imid),
            )

        # Samples visuels
        if len(samples) < args.n_samples:
            for b in range(I0.shape[0]):
                if len(samples) >= args.n_samples:
                    break
                samples.append({
                    "I0":    to_np(I0[b]),
                    "I1":    to_np(I1[b]),
                    "GT":    to_np(Imid[b]),
                    "avg":   to_np(pred_avg[b]),
                    "of":    to_np(pred_of[b]),
                    "px":    to_np(pred_px[b]),
                    "gan":   to_np(pred_gan[b]) if pred_gan is not None else None,
                    "psnr_avg": float(compute_psnr(pred_avg[b:b+1], Imid[b:b+1])),
                    "psnr_of":  float(compute_psnr(pred_of[b:b+1],  Imid[b:b+1])),
                    "psnr_px":  float(compute_psnr(pred_px[b:b+1],  Imid[b:b+1])),
                    "psnr_gan": float(compute_psnr(pred_gan[b:b+1], Imid[b:b+1])) if pred_gan is not None else None,
                })

        if (i + 1) % 20 == 0:
            m_of = acc_of.compute()
            m_px = acc_px.compute()
            print(f"  {i+1}/{args.n_eval} | OF={m_of.get('psnr',0):.2f} dB | PixelV2={m_px.get('psnr',0):.2f} dB")

    # Résultats
    m_avg = acc_avg.compute()
    m_of  = acc_of.compute()
    m_px  = acc_px.compute()
    m_gan = acc_gan.compute() if acc_gan._counts else {}

    rows = [
        ("Moyenne (I0+I1)/2",  m_avg),
        ("Optical Flow (Farneback)", m_of),
        ("UNetPixelV2",        m_px),
        ("UNetPixelV2 + GAN",  m_gan),
    ]

    print("\n" + "="*65)
    print(f"{'Modèle':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("="*65)
    for name, m in rows:
        if m:
            print(f"{name:<30} {m.get('psnr',0):>8.3f} {m.get('ssim',0):>8.4f} {m.get('lpips',0):>8.4f}")
    print("="*65)

    # ---- Comparaisons visuelles ----
    cols      = ["I₀", "I₁", "GT", "Moyenne", "Optical Flow\n(Farneback)", "UNetPixelV2", "UNetPixelV2\n+ GAN"]
    col_keys  = ["I0", "I1", "GT", "avg",     "of",                        "px",           "gan"]

    for si, s in enumerate(samples):
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.5))
        fig.suptitle(
            f"Sample {si+1}  |  Moyenne: {s['psnr_avg']:.2f} dB"
            f"  |  OF: {s['psnr_of']:.2f} dB"
            f"  |  PixelV2: {s['psnr_px']:.2f} dB"
            + (f"  |  GAN: {s['psnr_gan']:.2f} dB" if s["psnr_gan"] else ""),
            fontsize=9, fontweight="bold"
        )
        for ax, key, title in zip(axes, col_keys, cols):
            img = s.get(key)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#333")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="white")
            ax.set_title(title, fontsize=7.5)
            ax.axis("off")

        plt.tight_layout()
        out = os.path.join(args.out_dir, f"of_comparison_{si:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sauvegardé : {out}")

    # ---- Tableau complet ----
    fig_t, ax_t = plt.subplots(figsize=(10, 3.5))
    ax_t.axis("off")
    fig_t.suptitle("Comparaison complète — Vimeo-90K val (Optical Flow inclus)",
                   fontsize=12, fontweight="bold")

    table_data  = []
    col_labels  = ["Modèle", "PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]
    for name, m in rows:
        if m:
            table_data.append([name,
                                f"{m.get('psnr',0):.3f}",
                                f"{m.get('ssim',0):.4f}",
                                f"{m.get('lpips',0):.4f}"])

    table = ax_t.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.2)

    for j in range(4):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Surligner meilleures valeurs (PSNR et SSIM max, LPIPS min)
    for col_i, best_fn in [(1, max), (2, max), (3, min)]:
        try:
            vals = [float(r[col_i]) for r in table_data]
            best = best_fn(vals)
            for ri, v in enumerate(vals):
                if abs(v - best) < 1e-5:
                    table[ri + 1, col_i].set_facecolor("#a8e6a3")
        except Exception:
            pass

    # Surligner ligne Optical Flow en bleu clair
    of_row = next((i for i, r in enumerate(table_data) if "Optical" in r[0]), None)
    if of_row is not None:
        for j in range(4):
            cell = table[of_row + 1, j]
            if cell.get_facecolor() == (1, 1, 1, 1) or cell.get_facecolor()[3] == 0:
                cell.set_facecolor("#d6eaf8")

    plt.tight_layout()
    out_table = os.path.join(args.out_dir, "full_table.png")
    plt.savefig(out_table, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tableau sauvegardé : {out_table}")

    # ---- Bar chart ----
    bar_names  = [r[0] for r in rows if r[1]]
    bar_psnr   = [r[1].get("psnr", 0) for r in rows if r[1]]
    bar_colors = ["#aaaaaa", "#3498db", "#4a90d9", "#e8a838"][:len(bar_names)]

    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    bars = ax_b.bar(bar_names, bar_psnr, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_b.set_ylabel("PSNR (dB)", fontsize=11)
    ax_b.set_title("PSNR — Vimeo-90K val (avec Optical Flow baseline)",
                   fontsize=11, fontweight="bold")
    ax_b.set_ylim(bottom=min(bar_psnr) - 1.5 if bar_psnr else 20)
    for bar, v in zip(bars, bar_psnr):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.tick_params(axis="x", labelrotation=10, labelsize=8)
    plt.tight_layout()
    out_bar = os.path.join(args.out_dir, "full_psnr_bar.png")
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart sauvegardé : {out_bar}")

    import json
    with open(os.path.join(args.out_dir, "full_metrics.json"), "w") as f:
        json.dump({name: {k: float(v) for k, v in m.items()} for name, m in rows if m}, f, indent=2)


if __name__ == "__main__":
    main()
