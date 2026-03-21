"""
poster/03_model_comparison.py — Comparaison visuelle des 3 modèles

Pour N triplets Vimeo-90K :
  - UNetPixelV2 (pixel)
  - UNetPixelV2 + GAN fine-tune
  - DiffusionUNetV3 (DDIM 50 steps)

Génère :
  results/comparison_XXX.png   (N grilles côte à côte)
  results/metrics_table.png    (tableau comparatif)

Usage:
    cd poster
    python 03_model_comparison.py --data_root "D:/projet mumltimodal/vimeo_triplet"
    python 03_model_comparison.py --data_root "D:/projet mumltimodal/vimeo_triplet" --skip_diffusion
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2
from models.unet_latent import FrozenVAE
from models.diffusion_v3 import DiffusionUNetV3, CosineNoiseSchedule, DDIMSampler
from metrics import compute_psnr, compute_ssim, LPIPSMetric, MetricAccumulator


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
    print(f"  PixelV2 loaded: epoch={ckpt.get('epoch', '?')}, best_psnr={ckpt.get('best_psnr', '?'):.2f} dB")
    return model


def load_gan(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = UNetPixelV2(base_ch=64).to(device)   # GAN trained with base_ch=64
    model.load_state_dict(ckpt["G"])
    model.eval()
    print(f"  GAN loaded: epoch={ckpt.get('epoch', '?')}")
    return model


def load_diffusion_v3(ckpt_path, vae_id, T_inf, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = DiffusionUNetV3(latent_ch=4, base_ch=args_.get("base_ch", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    T         = args_.get("T", 1000)
    schedule  = CosineNoiseSchedule(T=T)
    sampler   = DDIMSampler(model, schedule, T_inf=T_inf, eta=0.0, prediction_type="v")
    sampler.timesteps = sampler.timesteps.to(device)
    schedule.to(device)

    print(f"  DiffusionV3 loaded: base_ch={args_.get('base_ch', 128)}, T_inf={T_inf}")

    vae = FrozenVAE(model_id=vae_id).to(device)
    return sampler, vae


# ---------------------------------------------------------------------------
# Tensor utils
# ---------------------------------------------------------------------------

def to_np_rgb(t):
    """[1,3,H,W] ou [3,H,W] dans [-1,1] → np [H,W,3] dans [0,1]."""
    if t.dim() == 4:
        t = t.squeeze(0)
    return ((t.cpu().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--pixel_ckpt",   type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--gan_ckpt",     type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2_gan/best.pth"))
    p.add_argument("--diff_ckpt",    type=str, default=os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth"))
    p.add_argument("--vae_id",       type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--T_inf",        type=int, default=50, help="DDIM steps (50 est suffisant pour la visualisation)")
    p.add_argument("--n_samples",    type=int, default=6,  help="Nombre de triplets à visualiser")
    p.add_argument("--n_eval",       type=int, default=100, help="Batches pour le tableau de métriques")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--skip_diffusion", action="store_true", help="Sauter le modèle de diffusion (long)")
    p.add_argument("--out_dir",      type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Data
    _, val_loader, _ = get_dataloaders(args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)

    # Models
    print("\nChargement des modèles...")
    pixel_model = load_pixel_v2(args.pixel_ckpt, device)
    gan_model   = load_gan(args.gan_ckpt, device) if os.path.isfile(args.gan_ckpt) else None
    diff_sampler, vae = (None, None)
    if not args.skip_diffusion and os.path.isfile(args.diff_ckpt):
        try:
            diff_sampler, vae = load_diffusion_v3(args.diff_ckpt, args.vae_id, args.T_inf, device)
        except Exception as e:
            print(f"  [!] Diffusion non chargé : {e}")

    # ---- Accumulate metrics ----
    acc_pixel = MetricAccumulator()
    acc_gan   = MetricAccumulator()
    acc_diff  = MetricAccumulator()
    acc_avg   = MetricAccumulator()
    lpips_fn  = LPIPSMetric().to(device)

    # ---- Collect visual samples ----
    samples = []   # list of (I0, I1, GT, pred_pixel, pred_gan, pred_diff)

    print(f"\nÉvaluation sur {args.n_eval} batches...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_eval:
                break

            I0   = batch["I0"].to(device)
            Imid = batch["Imid"].to(device)
            I1   = batch["I1"].to(device)

            # Predictions
            pred_pixel = pixel_model(I0, I1).clamp(-1, 1)
            pred_avg   = (0.5 * (I0 + I1)).clamp(-1, 1)

            pred_gan  = None
            pred_diff = None

            if gan_model is not None:
                pred_gan = gan_model(I0, I1).clamp(-1, 1)

            if diff_sampler is not None:
                z0 = vae.encode(I0)
                z1 = vae.encode(I1)
                z_pred    = diff_sampler.sample(z0, z1, device=device)
                pred_diff = vae.decode(z_pred)

            # Metrics
            acc_pixel.update(
                psnr=compute_psnr(pred_pixel, Imid),
                ssim=compute_ssim(pred_pixel.cpu(), Imid.cpu()),
                lpips=lpips_fn(pred_pixel, Imid),
            )
            acc_avg.update(
                psnr=compute_psnr(pred_avg, Imid),
                ssim=compute_ssim(pred_avg.cpu(), Imid.cpu()),
                lpips=lpips_fn(pred_avg, Imid),
            )
            if pred_gan is not None:
                acc_gan.update(
                    psnr=compute_psnr(pred_gan, Imid),
                    ssim=compute_ssim(pred_gan.cpu(), Imid.cpu()),
                    lpips=lpips_fn(pred_gan, Imid),
                )
            if pred_diff is not None:
                acc_diff.update(
                    psnr=compute_psnr(pred_diff, Imid),
                    ssim=compute_ssim(pred_diff.cpu(), Imid.cpu()),
                    lpips=lpips_fn(pred_diff, Imid),
                )

            # Collecte samples visuels
            if len(samples) < args.n_samples:
                for b in range(I0.shape[0]):
                    if len(samples) >= args.n_samples:
                        break
                    samples.append({
                        "I0":    to_np_rgb(I0[b]),
                        "I1":    to_np_rgb(I1[b]),
                        "GT":    to_np_rgb(Imid[b]),
                        "avg":   to_np_rgb(pred_avg[b]),
                        "pixel": to_np_rgb(pred_pixel[b]),
                        "gan":   to_np_rgb(pred_gan[b])   if pred_gan  is not None else None,
                        "diff":  to_np_rgb(pred_diff[b])  if pred_diff is not None else None,
                        "psnr_pixel": float(compute_psnr(pred_pixel[b:b+1], Imid[b:b+1])),
                        "psnr_gan":   float(compute_psnr(pred_gan[b:b+1],   Imid[b:b+1])) if pred_gan  is not None else None,
                        "psnr_diff":  float(compute_psnr(pred_diff[b:b+1],  Imid[b:b+1])) if pred_diff is not None else None,
                        "ssim_pixel": float(compute_ssim(pred_pixel[b:b+1].cpu(), Imid[b:b+1].cpu())),
                    })

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{args.n_eval} batches")

    # ---- Print metrics ----
    m_pixel = acc_pixel.compute()
    m_avg   = acc_avg.compute()
    m_gan   = acc_gan.compute()   if acc_gan._counts  else {}
    m_diff  = acc_diff.compute()  if acc_diff._counts else {}

    print("\n" + "="*65)
    print(f"{'Modèle':<30} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8}")
    print("="*65)
    rows = [
        ("Moyenne (baseline)",    m_avg),
        ("UNetPixelV2",           m_pixel),
        ("UNetPixelV2 + GAN",     m_gan),
        ("DiffusionUNetV3",       m_diff),
    ]
    for name, m in rows:
        if m:
            print(f"{name:<30} {m.get('psnr', float('nan')):>10.3f} {m.get('ssim', float('nan')):>8.4f} {m.get('lpips', float('nan')):>8.4f}")
    print("="*65)

    # ---- Figure 1 : grilles de comparaison visuelle ----
    has_gan  = any(s["gan"]  is not None for s in samples)
    has_diff = any(s["diff"] is not None for s in samples)

    cols    = ["I₀", "I₁", "GT (Imid)", "Moyenne", "PixelV2"]
    col_keys = ["I0", "I1", "GT", "avg", "pixel"]
    if has_gan:
        cols.append("PixelV2 + GAN")
        col_keys.append("gan")
    if has_diff:
        cols.append(f"Diffusion\n(DDIM {args.T_inf})")
        col_keys.append("diff")

    n_cols = len(cols)

    for si, s in enumerate(samples):
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
        fig.suptitle(
            f"Sample {si+1} — PixelV2: {s['psnr_pixel']:.2f} dB"
            + (f" | GAN: {s['psnr_gan']:.2f} dB" if s["psnr_gan"] else "")
            + (f" | Diff: {s['psnr_diff']:.2f} dB" if s["psnr_diff"] else ""),
            fontsize=10, fontweight="bold"
        )
        for ax, key, col_title in zip(axes, col_keys, cols):
            img = s.get(key)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#333333")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="white", fontsize=12)
            ax.set_title(col_title, fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        out = os.path.join(args.out_dir, f"comparison_{si:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sauvegardé : {out}")

    # ---- Figure 2 : tableau de métriques ----
    fig_t, ax_t = plt.subplots(figsize=(9, 3.5))
    ax_t.axis("off")
    fig_t.suptitle("Tableau comparatif — Vimeo-90K Val", fontsize=13, fontweight="bold")

    table_data = []
    col_labels = ["Modèle", "PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]
    for name, m in rows:
        if m:
            table_data.append([
                name,
                f"{m.get('psnr', float('nan')):.3f}",
                f"{m.get('ssim', float('nan')):.4f}",
                f"{m.get('lpips', float('nan')):.4f}",
            ])

    table = ax_t.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    # Colorier la meilleure valeur par colonne (PSNR et SSIM max, LPIPS min)
    if table_data:
        for col_i, best_fn in [(1, max), (2, max), (3, min)]:
            try:
                vals = [float(row[col_i]) for row in table_data]
                best = best_fn(vals)
                for row_i, v in enumerate(vals):
                    if abs(v - best) < 1e-6:
                        table[row_i + 1, col_i].set_facecolor("#d4f1c1")
            except Exception:
                pass

    plt.tight_layout()
    out_table = os.path.join(args.out_dir, "metrics_table.png")
    plt.savefig(out_table, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_table}")

    # ---- Figure 3 : bar chart PSNR ----
    fig_b, ax_b = plt.subplots(figsize=(7, 4))
    bar_names  = [n for n, m in rows if m]
    bar_psnr   = [m.get("psnr", 0) for _, m in rows if m]
    bar_colors = ["#aaa", "#4a90d9", "#e8a838", "#c0392b"][:len(bar_names)]

    bars = ax_b.bar(bar_names, bar_psnr, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_b.set_ylabel("PSNR (dB)", fontsize=11)
    ax_b.set_title("Comparaison PSNR — Vimeo-90K Val", fontsize=12, fontweight="bold")
    ax_b.set_ylim(bottom=min(bar_psnr) - 2 if bar_psnr else 25)
    for bar, v in zip(bars, bar_psnr):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.tick_params(axis="x", labelrotation=12, labelsize=9)

    plt.tight_layout()
    out_bar = os.path.join(args.out_dir, "psnr_bar.png")
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_bar}")

    # Sauvegarder métriques en JSON
    import json
    results = {}
    for name, m in rows:
        if m:
            results[name] = {k: float(v) for k, v in m.items()}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Métriques JSON : {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
