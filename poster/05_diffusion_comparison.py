"""
poster/05_diffusion_comparison.py — Diffusion pure vs Warm-start diffusion

Deux modes :
  A. Diffusion pure : z_T ~ N(0,I) → DDIM 50 steps → Imid_pred
     Avantage  : peut halluciner des détails réalistes
     Inconvénient : instable sur les hautes fréquences (PSNR plus bas)

  B. Warm-start diffusion : UNetPixelV2(I0,I1) → encode VAE → bruiter à t_start
                            → DDIM de t_start→0 → décoder
     Avantage  : part d'une bonne estimation, regagner des détails fins
     Inconvénient : si t_start trop bas → rien ne change ; trop haut → perd le contenu

Génère :
  results/diffusion_comparison_XXX.png  (grilles)
  results/diffusion_metrics_table.png
  results/diffusion_psnr_vs_tstart.png  (courbe PSNR en fonction de t_start)

Usage:
    cd poster
    python 05_diffusion_comparison.py --data_root "D:/projet mumltimodal/vimeo_triplet"
    python 05_diffusion_comparison.py --data_root "..." --steps 50 --t_start 200
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
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
    return model


def load_diffusion_v3(ckpt_path, vae_id, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = DiffusionUNetV3(latent_ch=4, base_ch=args_.get("base_ch", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    T        = args_.get("T", 1000)
    schedule = CosineNoiseSchedule(T=T)
    schedule.to(device)
    vae = FrozenVAE(model_id=vae_id).to(device)
    print(f"  DiffusionV3 loaded: epoch={ckpt.get('epoch','?')}, T={T}")
    return model, schedule, vae


# ---------------------------------------------------------------------------
# DDIM warm-start (v-prediction, DiffusionUNetV3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_warmstart_v(model, schedule, z_pixel, t_start, steps, z0, z1, device):
    """
    1. Bruiter z_pixel jusqu'au niveau t_start
    2. DDIM de t_start → 0 en `steps` pas (v-prediction)
    """
    ab = schedule["alpha_bar"].to(device)
    B  = z_pixel.shape[0]

    # Bruitagehttps
    t_tens = torch.full((B,), t_start, device=device, dtype=torch.long)
    noise  = torch.randn_like(z_pixel)
    sa = ab[t_start].sqrt()
    sb = (1 - ab[t_start]).sqrt()
    z_t = sa * z_pixel + sb * noise

    # Timesteps de t_start → 0
    timesteps = torch.linspace(t_start, 0, steps + 1, dtype=torch.long).to(device)
    ts_cur  = timesteps[:-1]
    ts_prev = timesteps[1:]

    z = z_t
    for t_cur, t_prev in zip(ts_cur, ts_prev):
        t_batch = torch.full((B,), t_cur.item(), device=device, dtype=torch.long)
        out = model(z, t_batch, z0, z1)   # v-prediction

        a_cur  = ab[t_cur]
        a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        # v-pred → x0, eps
        x0_pred = a_cur.sqrt() * z - (1 - a_cur).sqrt() * out
        eps_eq  = (1 - a_cur).sqrt() * z + a_cur.sqrt() * out
        x0_pred = x0_pred.clamp(-10, 10)

        dir_xt = (1 - a_prev).clamp(0).sqrt() * eps_eq
        z = a_prev.sqrt() * x0_pred + dir_xt

    return z


@torch.no_grad()
def ddim_pure_v(model, schedule, steps, z0, z1, device):
    """Diffusion pure : z_T ~ N(0,I) → DDIM steps → z_0 (v-prediction)."""
    ab = schedule["alpha_bar"].to(device)
    T  = schedule.T
    shape = z0.shape

    z = torch.randn(shape, device=device)
    timesteps = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long, device=device)
    ts_cur  = timesteps[:-1]
    ts_prev = timesteps[1:]

    for t_cur, t_prev in zip(ts_cur, ts_prev):
        t_batch = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.long)
        out = model(z, t_batch, z0, z1)  # v-prediction

        a_cur  = ab[t_cur]
        a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        x0_pred = a_cur.sqrt() * z - (1 - a_cur).sqrt() * out
        eps_eq  = (1 - a_cur).sqrt() * z + a_cur.sqrt() * out
        x0_pred = x0_pred.clamp(-10, 10)
        dir_xt  = (1 - a_prev).clamp(0).sqrt() * eps_eq
        z = a_prev.sqrt() * x0_pred + dir_xt

    return z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    p.add_argument("--diff_ckpt",  type=str, default=os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth"))
    p.add_argument("--vae_id",     type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--steps",      type=int, default=50,  help="Nombre de steps DDIM")
    p.add_argument("--t_start",    type=int, default=200, help="Niveau de bruit pour le warm-start (0-999)")
    p.add_argument("--n_samples",  type=int, default=5,   help="Triplets visualisés")
    p.add_argument("--n_eval",     type=int, default=50,  help="Batches pour les métriques")
    p.add_argument("--batch_size", type=int, default=2)
    # Courbe t_start : évaluer pour plusieurs valeurs de t_start
    p.add_argument("--sweep_tstart", action="store_true",
                   help="Évaluer plusieurs t_start (100,200,300,400,500) — lent")
    p.add_argument("--out_dir",    type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def run_eval(model_diff, schedule, vae, pixel_model, loader, device,
             steps, t_start, n_eval, n_samples):
    """Évalue les 3 méthodes (pixel, diffusion pure, warm-start) et collecte les samples."""

    schedule.to(device)

    acc_pixel = MetricAccumulator()
    acc_pure  = MetricAccumulator()
    acc_warm  = MetricAccumulator()
    acc_avg   = MetricAccumulator()
    lpips_fn  = LPIPSMetric().to(device)

    samples = []

    for i, batch in enumerate(loader):
        if i >= n_eval:
            break

        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        # Pixel baseline
        with torch.no_grad():
            pred_pixel = pixel_model(I0, I1).clamp(-1, 1)
            pred_avg   = (0.5 * (I0 + I1)).clamp(-1, 1)

        # Encode
        z0     = vae.encode(I0)
        z1     = vae.encode(I1)
        z_pixel = vae.encode(pred_pixel)

        # Diffusion pure (bruit total → 0)
        with torch.no_grad():
            z_pure = ddim_pure_v(model_diff, schedule, steps, z0, z1, device)
            pred_pure = vae.decode(z_pure)

        # Warm-start (bruiter z_pixel à t_start, débruiter)
        with torch.no_grad():
            z_warm = ddim_warmstart_v(model_diff, schedule, z_pixel, t_start, steps, z0, z1, device)
            pred_warm = vae.decode(z_warm)

        # Métriques
        acc_pixel.update(psnr=compute_psnr(pred_pixel, Imid),
                         ssim=compute_ssim(pred_pixel.cpu(), Imid.cpu()),
                         lpips=lpips_fn(pred_pixel, Imid))
        acc_avg.update(psnr=compute_psnr(pred_avg, Imid),
                       ssim=compute_ssim(pred_avg.cpu(), Imid.cpu()),
                       lpips=lpips_fn(pred_avg, Imid))
        acc_pure.update(psnr=compute_psnr(pred_pure, Imid),
                        ssim=compute_ssim(pred_pure.cpu(), Imid.cpu()),
                        lpips=lpips_fn(pred_pure, Imid))
        acc_warm.update(psnr=compute_psnr(pred_warm, Imid),
                        ssim=compute_ssim(pred_warm.cpu(), Imid.cpu()),
                        lpips=lpips_fn(pred_warm, Imid))

        # Samples visuels
        if len(samples) < n_samples:
            for b in range(I0.shape[0]):
                if len(samples) >= n_samples:
                    break
                samples.append({
                    "I0":    to_np(I0[b]),
                    "I1":    to_np(I1[b]),
                    "GT":    to_np(Imid[b]),
                    "avg":   to_np(pred_avg[b]),
                    "pixel": to_np(pred_pixel[b]),
                    "pure":  to_np(pred_pure[b]),
                    "warm":  to_np(pred_warm[b]),
                    "psnr_pixel": float(compute_psnr(pred_pixel[b:b+1], Imid[b:b+1])),
                    "psnr_pure":  float(compute_psnr(pred_pure[b:b+1],  Imid[b:b+1])),
                    "psnr_warm":  float(compute_psnr(pred_warm[b:b+1],  Imid[b:b+1])),
                    "ssim_pixel": float(compute_ssim(pred_pixel[b:b+1].cpu(), Imid[b:b+1].cpu())),
                    "ssim_warm":  float(compute_ssim(pred_warm[b:b+1].cpu(),  Imid[b:b+1].cpu())),
                })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_eval} batches | pixel={acc_pixel.compute().get('psnr',0):.2f}"
                  f" | pure={acc_pure.compute().get('psnr',0):.2f}"
                  f" | warm={acc_warm.compute().get('psnr',0):.2f}")

    return (acc_avg.compute(), acc_pixel.compute(),
            acc_pure.compute(), acc_warm.compute(), samples)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | steps={args.steps} | t_start={args.t_start}")

    # Data
    _, val_loader, _ = get_dataloaders(
        args.data_root, crop_size=256, batch_size=args.batch_size, num_workers=0)

    # Models
    print("\nChargement des modèles...")
    pixel_model = load_pixel_v2(args.pixel_ckpt, device)
    model_diff, schedule, vae = load_diffusion_v3(args.diff_ckpt, args.vae_id, device)

    # ---- Évaluation principale ----
    print(f"\nÉvaluation : {args.n_eval} batches, steps={args.steps}, t_start={args.t_start}")
    m_avg, m_pixel, m_pure, m_warm, samples = run_eval(
        model_diff, schedule, vae, pixel_model, val_loader, device,
        args.steps, args.t_start, args.n_eval, args.n_samples
    )

    # Print
    rows = [
        ("Moyenne (baseline)", m_avg),
        ("UNetPixelV2",        m_pixel),
        (f"Diffusion pure ({args.steps} steps)", m_pure),
        (f"Warm-start t={args.t_start} ({args.steps} steps)", m_warm),
    ]
    print("\n" + "="*70)
    print(f"{'Modèle':<40} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("="*70)
    for name, m in rows:
        if m:
            print(f"{name:<40} {m.get('psnr', 0):>8.3f} {m.get('ssim', 0):>8.4f} {m.get('lpips', 0):>8.4f}")
    print("="*70)

    # ---- Figures de comparaison visuelle ----
    cols    = ["I₀", "I₁", "GT", "Moyenne", "PixelV2",
               f"Diff. pure\n({args.steps}st)", f"Warm-start\nt={args.t_start}"]
    col_keys = ["I0", "I1", "GT", "avg", "pixel", "pure", "warm"]

    for si, s in enumerate(samples):
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.5))
        fig.suptitle(
            f"Sample {si+1} — PixelV2: {s['psnr_pixel']:.2f} dB "
            f"| Pure: {s['psnr_pure']:.2f} dB "
            f"| Warm-start(t={args.t_start}): {s['psnr_warm']:.2f} dB",
            fontsize=9, fontweight="bold"
        )
        for ax, key, title in zip(axes, col_keys, cols):
            ax.imshow(s[key])
            ax.set_title(title, fontsize=7)
            ax.axis("off")
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"diffusion_comparison_{si:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sauvegardé : {out}")

    # ---- Tableau de métriques ----
    fig_t, ax_t = plt.subplots(figsize=(11, 3.5))
    ax_t.axis("off")
    fig_t.suptitle(
        f"Métriques — Vimeo-90K val | steps={args.steps} | t_start={args.t_start}",
        fontsize=12, fontweight="bold"
    )
    table_data = []
    col_labels = ["Modèle", "PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]
    for name, m in rows:
        if m:
            table_data.append([
                name,
                f"{m.get('psnr', 0):.3f}",
                f"{m.get('ssim', 0):.4f}",
                f"{m.get('lpips', 0):.4f}",
            ])
    table = ax_t.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    # Colorier meilleures valeurs
    if table_data:
        for col_i, best_fn in [(1, max), (2, max), (3, min)]:
            try:
                vals = [float(r[col_i]) for r in table_data]
                best = best_fn(vals)
                for ri, v in enumerate(vals):
                    if abs(v - best) < 1e-6:
                        table[ri + 1, col_i].set_facecolor("#d4f1c1")
            except Exception:
                pass
    plt.tight_layout()
    out_table = os.path.join(args.out_dir, "diffusion_metrics_table.png")
    plt.savefig(out_table, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_table}")

    # ---- Bar chart PSNR ----
    bar_names  = [r[0] for r in rows if r[1]]
    bar_psnr   = [r[1].get("psnr", 0) for r in rows if r[1]]
    bar_colors = ["#aaaaaa", "#4a90d9", "#c0392b", "#27ae60"][:len(bar_names)]

    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    bars = ax_b.bar(bar_names, bar_psnr, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_b.set_ylabel("PSNR (dB)", fontsize=11)
    ax_b.set_title(f"Comparaison PSNR — diffusion pure vs warm-start (t_start={args.t_start})",
                   fontsize=11, fontweight="bold")
    ax_b.set_ylim(bottom=min(bar_psnr) - 2 if bar_psnr else 20)
    for bar, v in zip(bars, bar_psnr):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.tick_params(axis="x", labelrotation=10, labelsize=8)
    plt.tight_layout()
    out_bar = os.path.join(args.out_dir, "diffusion_psnr_bar.png")
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_bar}")

    # ---- Optionnel : sweep sur t_start ----
    if args.sweep_tstart:
        t_starts = [50, 100, 150, 200, 300, 400, 500]
        psnr_warm_list = []
        ssim_warm_list = []

        print(f"\nSweep t_start : {t_starts} (steps={args.steps}, n_eval=20)")
        for ts in t_starts:
            acc_w = MetricAccumulator()
            lp    = LPIPSMetric().to(device)
            for i, batch in enumerate(val_loader):
                if i >= 20:
                    break
                I0   = batch["I0"].to(device)
                Imid = batch["Imid"].to(device)
                I1   = batch["I1"].to(device)
                with torch.no_grad():
                    pred_pixel = pixel_model(I0, I1).clamp(-1, 1)
                    z0     = vae.encode(I0)
                    z1     = vae.encode(I1)
                    z_px   = vae.encode(pred_pixel)
                    z_w    = ddim_warmstart_v(model_diff, schedule, z_px, ts, args.steps, z0, z1, device)
                    pred_w = vae.decode(z_w)
                acc_w.update(psnr=compute_psnr(pred_w, Imid),
                             ssim=compute_ssim(pred_w.cpu(), Imid.cpu()))
            m = acc_w.compute()
            psnr_warm_list.append(m.get("psnr", 0))
            ssim_warm_list.append(m.get("ssim", 0))
            print(f"  t_start={ts:4d} -> PSNR={m.get('psnr',0):.3f} dB | SSIM={m.get('ssim',0):.4f}")

        # Ligne de référence : PixelV2
        psnr_pixel_ref = m_pixel.get("psnr", 0)

        fig_sw, axes_sw = plt.subplots(1, 2, figsize=(12, 4))
        fig_sw.suptitle(
            f"Effet du niveau de bruit t_start sur le warm-start (DDIM {args.steps} steps)",
            fontsize=12, fontweight="bold"
        )
        ax1, ax2 = axes_sw

        ax1.plot(t_starts, psnr_warm_list, "o-", color="#27ae60", linewidth=2, label="Warm-start")
        ax1.axhline(psnr_pixel_ref, color="#4a90d9", linestyle="--", linewidth=1.5,
                    label=f"UNetPixelV2 ({psnr_pixel_ref:.2f} dB)")
        ax1.set_xlabel("t_start (niveau de bruit)", fontsize=10)
        ax1.set_ylabel("PSNR (dB)", fontsize=10)
        ax1.set_title("PSNR vs t_start", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_starts, ssim_warm_list, "s-", color="#e67e22", linewidth=2, label="Warm-start")
        ax2.set_xlabel("t_start (niveau de bruit)", fontsize=10)
        ax2.set_ylabel("SSIM", fontsize=10)
        ax2.set_title("SSIM vs t_start", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_sweep = os.path.join(args.out_dir, "diffusion_psnr_vs_tstart.png")
        plt.savefig(out_sweep, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sauvegardé : {out_sweep}")

    # Sauvegarder métriques
    import json
    results = {}
    for name, m in rows:
        if m:
            results[name] = {k: float(v) for k, v in m.items()}
    out_json = os.path.join(args.out_dir, "diffusion_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON : {out_json}")


if __name__ == "__main__":
    main()
