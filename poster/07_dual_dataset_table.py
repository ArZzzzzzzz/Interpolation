"""
poster/07_dual_dataset_table.py — Tableau comparatif sur 2 datasets

Pour chaque modèle, évalue sur :
  - Vimeo-90K val  (dataset d'entraînement référence)
  - Triplets vidéo (D:/vidéo/triplets, dataset "terrain réel")

Génère :
  results/dual_table.png   — tableau avec 2 colonnes par métrique

Usage:
    cd poster
    python 07_dual_dataset_table.py
    python 07_dual_dataset_table.py --n_eval 80 --video_root "D:/vidéo/triplets"
"""

import argparse
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader, Subset
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2
from models.unet_latent import FrozenVAE
from models.diffusion_v3 import DiffusionUNetV3, CosineNoiseSchedule
from metrics import compute_psnr, compute_ssim, LPIPSMetric, MetricAccumulator

sys.path.insert(0, os.path.join(ROOT, "gan"))
from extracted_dataset import ExtractedTripletDataset


# ---------------------------------------------------------------------------
# DDIM inline (v-prediction, pas de bug CPU/CUDA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_pure(model, schedule, steps, z0, z1, device):
    ab = schedule["alpha_bar"].to(device)
    T  = schedule.T
    shape = z0.shape
    z = torch.randn(shape, device=device)
    ts = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long, device=device)
    for t_cur, t_prev in zip(ts[:-1], ts[1:]):
        t_b = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.long)
        out = model(z, t_b, z0, z1)
        a_c = ab[t_cur]
        a_p = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        x0  = (a_c.sqrt() * z - (1 - a_c).sqrt() * out).clamp(-10, 10)
        eps = (1 - a_c).sqrt() * z + a_c.sqrt() * out
        z   = a_p.sqrt() * x0 + (1 - a_p).clamp(0).sqrt() * eps
    return z


@torch.no_grad()
def ddim_warmstart(model, schedule, z_pixel, t_start, steps, z0, z1, device):
    ab = schedule["alpha_bar"].to(device)
    B  = z_pixel.shape[0]
    sa = ab[t_start].sqrt()
    sb = (1 - ab[t_start]).sqrt()
    z  = sa * z_pixel + sb * torch.randn_like(z_pixel)
    ts = torch.linspace(t_start, 0, steps + 1, dtype=torch.long, device=device)
    for t_cur, t_prev in zip(ts[:-1], ts[1:]):
        t_b = torch.full((B,), t_cur.item(), device=device, dtype=torch.long)
        out = model(z, t_b, z0, z1)
        a_c = ab[t_cur]
        a_p = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        x0  = (a_c.sqrt() * z - (1 - a_c).sqrt() * out).clamp(-10, 10)
        eps = (1 - a_c).sqrt() * z + a_c.sqrt() * out
        z   = a_p.sqrt() * x0 + (1 - a_p).clamp(0).sqrt() * eps
    return z


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


def load_diffusion_v3(ckpt_path, vae_id, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = DiffusionUNetV3(latent_ch=4, base_ch=args_.get("base_ch", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    schedule = CosineNoiseSchedule(T=args_.get("T", 1000))
    schedule.to(device)
    vae = FrozenVAE(model_id=vae_id).to(device)
    return model, schedule, vae


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(loader, pixel_model, gan_model, diff_model, schedule, vae,
                    device, n_eval, steps, t_start, skip_diffusion=False):
    acc = {
        "avg":   MetricAccumulator(),
        "pixel": MetricAccumulator(),
        "gan":   MetricAccumulator(),
        "pure":  MetricAccumulator(),
        "warm":  MetricAccumulator(),
    }
    lpips_fn = LPIPSMetric().to(device)

    for i, batch in enumerate(loader):
        if i >= n_eval:
            break

        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        # Clip pour être sûr que c'est dans [-1,1]
        I0   = I0.clamp(-1, 1)
        I1   = I1.clamp(-1, 1)
        Imid = Imid.clamp(-1, 1)

        pred_avg   = (0.5 * (I0 + I1)).clamp(-1, 1)
        pred_pixel = pixel_model(I0, I1).clamp(-1, 1)
        pred_gan   = gan_model(I0, I1).clamp(-1, 1)

        for key, pred in [("avg", pred_avg), ("pixel", pred_pixel), ("gan", pred_gan)]:
            acc[key].update(
                psnr  = compute_psnr(pred, Imid),
                ssim  = compute_ssim(pred.cpu(), Imid.cpu()),
                lpips = lpips_fn(pred, Imid),
            )

        if not skip_diffusion:
            z0     = vae.encode(I0)
            z1     = vae.encode(I1)
            z_px   = vae.encode(pred_pixel)

            z_pure = ddim_pure(diff_model, schedule, steps, z0, z1, device)
            pred_pure = vae.decode(z_pure)

            z_warm = ddim_warmstart(diff_model, schedule, z_px, t_start, steps, z0, z1, device)
            pred_warm = vae.decode(z_warm)

            for key, pred in [("pure", pred_pure), ("warm", pred_warm)]:
                acc[key].update(
                    psnr  = compute_psnr(pred, Imid),
                    ssim  = compute_ssim(pred.cpu(), Imid.cpu()),
                    lpips = lpips_fn(pred, Imid),
                )

        if (i + 1) % 20 == 0:
            pxl = acc["pixel"].compute().get("psnr", 0)
            print(f"    {i+1}/{n_eval} | pixel={pxl:.2f} dB", end="\r")

    print()
    return {k: v.compute() if v._counts else {} for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Table figure
# ---------------------------------------------------------------------------

def make_dual_table(results_vimeo, results_video, model_labels,
                    model_keys, steps, t_start, out_path, skip_diffusion):

    metrics = ["PSNR (dB)", "SSIM", "LPIPS"]
    metric_keys = ["psnr", "ssim", "lpips"]
    higher_better = [True, True, False]

    # Build data matrix : rows = models, cols = (Vimeo PSNR, Video PSNR, Vimeo SSIM, ...)
    col_headers = []
    for m in metrics:
        col_headers.append(f"{m}\nVimeo-90K")
        col_headers.append(f"{m}\nVidéos")

    rows_data = []
    for key, label in zip(model_keys, model_labels):
        rv = results_vimeo.get(key, {})
        rk = results_video.get(key, {})
        row = [label]
        for mk in metric_keys:
            v_vimeo = rv.get(mk, float("nan"))
            v_video = rk.get(mk, float("nan"))
            row.append(f"{v_vimeo:.3f}" if not np.isnan(v_vimeo) else "—")
            row.append(f"{v_video:.3f}"  if not np.isnan(v_video)  else "—")
        rows_data.append(row)

    n_cols = 1 + len(metrics) * 2
    n_rows = len(rows_data)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, n_rows * 0.7 + 2.0))
    ax.axis("off")
    fig.suptitle(
        f"Évaluation sur 2 datasets — Vimeo-90K val vs Triplets vidéo réelle\n"
        f"(DDIM steps={steps}, warm-start t={t_start})",
        fontsize=12, fontweight="bold", y=0.98
    )

    col_labels_full = ["Modèle"] + col_headers
    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels_full,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 2.2)

    # Style en-tête
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Colonnes Vimeo (pair) vs Vidéo (impair) — légère distinction de fond
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols):
            is_video_col = (j % 2 == 0)  # 1=Vimeo, 2=Vidéo, 3=Vimeo, ...
            cell = table[i, j]
            if is_video_col:
                cell.set_facecolor("#f0f7ff")
            else:
                cell.set_facecolor("#ffffff")

    # Colorier la meilleure valeur dans chaque groupe de colonnes
    for m_i, (mk, hb) in enumerate(zip(metric_keys, higher_better)):
        for ds_i in range(2):   # 0=Vimeo, 1=Video
            col_j = 1 + m_i * 2 + ds_i
            vals = []
            for row_i, (key, _) in enumerate(zip(model_keys, model_labels)):
                ds = results_vimeo if ds_i == 0 else results_video
                v = ds.get(key, {}).get(mk, float("nan"))
                vals.append(v)

            if all(np.isnan(v) for v in vals):
                continue
            best_val = (max if hb else min)(v for v in vals if not np.isnan(v))
            for row_i, v in enumerate(vals):
                if not np.isnan(v) and abs(v - best_val) < 1e-5:
                    table[row_i + 1, col_j].set_facecolor("#a8e6a3")

    # Légende dataset
    handles = [
        mpatches.Patch(facecolor="#ffffff", edgecolor="gray", label="Vimeo-90K val"),
        mpatches.Patch(facecolor="#f0f7ff", edgecolor="gray", label="Triplets vidéo (D:/vidéo)"),
        mpatches.Patch(facecolor="#a8e6a3", edgecolor="gray", label="Meilleur par colonne"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tableau sauvegardé : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--video_root",  type=str, default="D:/vidéo/triplets")
    p.add_argument("--pixel_ckpt",  type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--gan_ckpt",    type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2_gan/best.pth"))
    p.add_argument("--diff_ckpt",   type=str, default=os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth"))
    p.add_argument("--vae_id",      type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--steps",       type=int, default=50)
    p.add_argument("--t_start",     type=int, default=200)
    p.add_argument("--n_eval",      type=int, default=80)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--skip_diffusion", action="store_true")
    p.add_argument("--out_dir",     type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Modèles
    print("\nChargement des modèles...")
    pixel_model = load_pixel_v2(args.pixel_ckpt, device)
    print("  PixelV2 OK")
    gan_model   = load_gan(args.gan_ckpt, device)
    print("  GAN OK")
    diff_model, schedule, vae = (None, None, None)
    if not args.skip_diffusion:
        diff_model, schedule, vae = load_diffusion_v3(args.diff_ckpt, args.vae_id, device)
        print("  DiffusionV3 + VAE OK")

    # Datasets
    print("\nChargement des datasets...")
    _, val_loader, _ = get_dataloaders(
        args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)

    video_ds     = ExtractedTripletDataset(args.video_root, augment=False)
    # Sous-échantillonnage aléatoire reproductible
    indices      = random.Random(42).sample(range(len(video_ds)), min(len(video_ds), args.n_eval * args.batch_size * 2))
    video_sub    = Subset(video_ds, indices)
    video_loader = DataLoader(video_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_keys   = ["avg", "pixel", "gan", "pure", "warm"]
    model_labels = [
        "Moyenne (I0+I1)/2",
        "UNetPixelV2",
        "UNetPixelV2 + GAN",
        f"Diffusion pure ({args.steps} steps)",
        f"Warm-start t={args.t_start} ({args.steps} steps)",
    ]
    if args.skip_diffusion:
        model_keys   = model_keys[:3]
        model_labels = model_labels[:3]

    # Éval Vimeo-90K
    print(f"\n[1/2] Éval Vimeo-90K val ({args.n_eval} batches)...")
    results_vimeo = evaluate_loader(
        val_loader, pixel_model, gan_model, diff_model, schedule, vae,
        device, args.n_eval, args.steps, args.t_start, args.skip_diffusion
    )

    # Éval vidéo
    print(f"\n[2/2] Éval triplets vidéo ({args.n_eval} batches)...")
    results_video = evaluate_loader(
        video_loader, pixel_model, gan_model, diff_model, schedule, vae,
        device, args.n_eval, args.steps, args.t_start, args.skip_diffusion
    )

    # Print console
    print("\n" + "="*80)
    print(f"{'Modèle':<35} {'PSNR Vim':>9} {'PSNR Vid':>9} {'SSIM Vim':>9} {'SSIM Vid':>9} {'LPIPS Vim':>10} {'LPIPS Vid':>10}")
    print("="*80)
    for key, label in zip(model_keys, model_labels):
        rv = results_vimeo.get(key, {})
        rk = results_video.get(key, {})
        print(f"{label:<35}"
              f" {rv.get('psnr', float('nan')):>9.3f}"
              f" {rk.get('psnr', float('nan')):>9.3f}"
              f" {rv.get('ssim', float('nan')):>9.4f}"
              f" {rk.get('ssim', float('nan')):>9.4f}"
              f" {rv.get('lpips', float('nan')):>10.4f}"
              f" {rk.get('lpips', float('nan')):>10.4f}")
    print("="*80)

    # Tableau figure
    out_table = os.path.join(args.out_dir, "dual_table.png")
    make_dual_table(results_vimeo, results_video, model_labels, model_keys,
                    args.steps, args.t_start, out_table, args.skip_diffusion)

    # JSON
    out_json = os.path.join(args.out_dir, "dual_metrics.json")
    with open(out_json, "w") as f:
        json.dump({"vimeo": {k: {mk: float(v) for mk, v in m.items()} for k, m in results_vimeo.items()},
                   "video": {k: {mk: float(v) for mk, v in m.items()} for k, m in results_video.items()}},
                  f, indent=2)
    print(f"JSON : {out_json}")


if __name__ == "__main__":
    main()
