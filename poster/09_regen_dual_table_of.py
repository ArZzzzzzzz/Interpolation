"""
poster/09_regen_dual_table_of.py — Régénère dual_table avec Optical Flow comme baseline

Charge dual_metrics.json (résultats pixel/gan/diffusion déjà calculés),
calcule uniquement l'optical flow sur les 2 datasets, et régénère le tableau.

Usage:
    cd poster
    python 09_regen_dual_table_of.py
    python 09_regen_dual_table_of.py --n_eval 80 --video_root "D:/vidéo/triplets"
"""

import argparse
import os
import sys
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from metrics import compute_psnr, compute_ssim, LPIPSMetric

sys.path.insert(0, os.path.join(ROOT, "gan"))
from extracted_dataset import ExtractedTripletDataset


# ---------------------------------------------------------------------------
# Optical flow helpers
# ---------------------------------------------------------------------------

def compute_farneback_flow(I0_np255, I1_np255):
    g0 = cv2.cvtColor(I0_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(I1_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_tensor(img_t, flow_np, t=0.5):
    H, W = img_t.shape[-2:]
    flow_t = torch.from_numpy(flow_np).float()
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    flow_x = t * flow_t[..., 0] / (W / 2)
    flow_y = t * flow_t[..., 1] / (H / 2)
    grid = torch.stack([gx + flow_x, gy + flow_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def tensor_to_np255(t):
    t = t.detach().cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def eval_of_on_loader(loader, n_eval, device):
    lpips_fn = LPIPSMetric().to(device)
    psnr_list, ssim_list, lpips_list = [], [], []

    for i, batch in enumerate(loader):
        if i >= n_eval:
            break

        I0   = batch["I0"].clamp(-1, 1)
        Imid = batch["Imid"].clamp(-1, 1)
        I1   = batch["I1"].clamp(-1, 1)

        I0_np = tensor_to_np255(I0)
        I1_np = tensor_to_np255(I1)

        # Optical flow image par image (CPU)
        preds = []
        for b in range(I0.shape[0]):
            flow = compute_farneback_flow(I0_np[b], I1_np[b])
            w0   = warp_tensor(I0[b:b+1], flow,  t= 0.5)
            w1   = warp_tensor(I1[b:b+1], flow,  t=-0.5)
            preds.append((0.5 * w0 + 0.5 * w1).clamp(-1, 1))
        pred = torch.cat(preds, dim=0)

        pred_d   = pred.to(device)
        Imid_d   = Imid.to(device)
        psnr_list.append(compute_psnr(pred_d, Imid_d))
        ssim_list.append(compute_ssim(pred, Imid))
        lpips_list.append(lpips_fn(pred_d, Imid_d))

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_eval} | OF PSNR={np.mean(psnr_list):.2f} dB", end="\r")

    print()
    return {
        "psnr":  float(np.mean(psnr_list)),
        "ssim":  float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
    }


# ---------------------------------------------------------------------------
# Table figure (identique à 07, avec la clé "of" pour Optical Flow)
# ---------------------------------------------------------------------------

def make_dual_table(results_vimeo, results_video, model_labels, model_keys, out_path):
    metrics      = ["PSNR (dB)", "SSIM", "LPIPS"]
    metric_keys  = ["psnr", "ssim", "lpips"]
    higher_better = [True, True, False]

    col_headers = []
    for m in metrics:
        col_headers.append(f"{m}\nVimeo-90K")
        col_headers.append(f"{m}\nVideos")

    rows_data = []
    for key, label in zip(model_keys, model_labels):
        rv = results_vimeo.get(key, {})
        rk = results_video.get(key, {})
        row = [label]
        for mk in metric_keys:
            v_vimeo = rv.get(mk, float("nan"))
            v_video = rk.get(mk, float("nan"))
            row.append(f"{v_vimeo:.3f}" if not np.isnan(v_vimeo) else "-")
            row.append(f"{v_video:.3f}"  if not np.isnan(v_video)  else "-")
        rows_data.append(row)

    n_cols = 1 + len(metrics) * 2
    n_rows = len(rows_data)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, n_rows * 0.7 + 2.0))
    ax.axis("off")
    fig.suptitle(
        "Evaluation sur 2 datasets — Vimeo-90K val vs Triplets video reelle",
        fontsize=12, fontweight="bold", y=0.98
    )

    col_labels_full = ["Modele"] + col_headers
    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels_full,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 2.2)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols):
            is_video_col = (j % 2 == 0)
            table[i, j].set_facecolor("#f0f7ff" if is_video_col else "#ffffff")

    for m_i, (mk, hb) in enumerate(zip(metric_keys, higher_better)):
        for ds_i in range(2):
            col_j = 1 + m_i * 2 + ds_i
            vals = []
            for key in model_keys:
                ds = results_vimeo if ds_i == 0 else results_video
                v = ds.get(key, {}).get(mk, float("nan"))
                vals.append(v)
            if all(np.isnan(v) for v in vals):
                continue
            best_val = (max if hb else min)(v for v in vals if not np.isnan(v))
            for row_i, v in enumerate(vals):
                if not np.isnan(v) and abs(v - best_val) < 1e-5:
                    table[row_i + 1, col_j].set_facecolor("#a8e6a3")

    handles = [
        mpatches.Patch(facecolor="#ffffff", edgecolor="gray", label="Vimeo-90K val"),
        mpatches.Patch(facecolor="#f0f7ff", edgecolor="gray", label="Triplets video (D:/video)"),
        mpatches.Patch(facecolor="#a8e6a3", edgecolor="gray", label="Meilleur par colonne"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tableau sauvegarde : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--video_root", type=str, default="D:/video/triplets")
    p.add_argument("--n_eval",     type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--json_in",    type=str,
                   default=os.path.join(os.path.dirname(__file__), "results/dual_metrics.json"))
    p.add_argument("--out_dir",    type=str,
                   default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Charger résultats existants (pixel, gan, diffusion)
    with open(args.json_in) as f:
        saved = json.load(f)
    results_vimeo = saved["vimeo"]   # keys: avg, pixel, gan, (pure, warm)
    results_video = saved["video"]

    # Calculer OF sur Vimeo-90K val
    print(f"\n[1/2] Optical Flow sur Vimeo-90K val ({args.n_eval} batches)...")
    _, val_loader, _ = get_dataloaders(
        args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)
    of_vimeo = eval_of_on_loader(val_loader, args.n_eval, device)
    print(f"  OF Vimeo -> PSNR={of_vimeo['psnr']:.3f} | SSIM={of_vimeo['ssim']:.4f} | LPIPS={of_vimeo['lpips']:.4f}")

    # Calculer OF sur triplets vidéo
    print(f"\n[2/2] Optical Flow sur triplets video ({args.n_eval} batches)...")
    video_ds  = ExtractedTripletDataset(args.video_root, augment=False)
    indices   = random.Random(42).sample(range(len(video_ds)), min(len(video_ds), args.n_eval * args.batch_size * 2))
    video_sub = Subset(video_ds, indices)
    video_loader = DataLoader(video_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
    of_video = eval_of_on_loader(video_loader, args.n_eval, device)
    print(f"  OF Video  -> PSNR={of_video['psnr']:.3f} | SSIM={of_video['ssim']:.4f} | LPIPS={of_video['lpips']:.4f}")

    # Injecter OF dans les résultats (remplace avg)
    results_vimeo["of"] = of_vimeo
    results_video["of"] = of_video

    # Ordre et labels pour le tableau
    has_diffusion = "pure" in results_vimeo
    model_keys   = ["of", "avg", "pixel", "gan"]
    model_labels = [
        "Optical Flow (Farneback)",
        "Moyenne (I0+I1)/2",
        "UNetPixelV2",
        "UNetPixelV2 + GAN",
    ]
    if has_diffusion:
        model_keys   += ["pure", "warm"]
        model_labels += ["Diffusion pure", "Warm-start diffusion"]

    # Print console
    print("\n" + "="*90)
    print(f"{'Modele':<30} {'PSNR Vim':>9} {'PSNR Vid':>9} {'SSIM Vim':>9} {'SSIM Vid':>9} {'LPIPS Vim':>10} {'LPIPS Vid':>10}")
    print("="*90)
    for key, label in zip(model_keys, model_labels):
        rv = results_vimeo.get(key, {})
        rk = results_video.get(key, {})
        print(f"{label:<30}"
              f" {rv.get('psnr', float('nan')):>9.3f}"
              f" {rk.get('psnr', float('nan')):>9.3f}"
              f" {rv.get('ssim', float('nan')):>9.4f}"
              f" {rk.get('ssim', float('nan')):>9.4f}"
              f" {rv.get('lpips', float('nan')):>10.4f}"
              f" {rk.get('lpips', float('nan')):>10.4f}")
    print("="*90)

    # Tableau figure
    out_table = os.path.join(args.out_dir, "dual_table.png")
    make_dual_table(results_vimeo, results_video, model_labels, model_keys, out_table)

    # Sauvegarder JSON mis à jour
    out_json = os.path.join(args.out_dir, "dual_metrics.json")
    with open(out_json, "w") as f:
        json.dump({"vimeo": results_vimeo, "video": results_video}, f, indent=2)
    print(f"JSON mis a jour : {out_json}")


if __name__ == "__main__":
    main()
