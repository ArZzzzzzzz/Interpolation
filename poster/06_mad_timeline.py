"""
poster/06_mad_timeline.py — MAD inter-frames en fonction du temps (courbe temporelle)

Montre sur quelle partie de la vidéo le modèle va avoir du mal :
- Courbe MAD(t) avec les pics
- Seuil horizontal (par défaut = 10) au-delà duquel la prédiction devient floue
- Zones "difficiles" colorées en rouge

Usage:
    cd poster
    python 06_mad_timeline.py --video "../Acedezinzin.mp4"
    python 06_mad_timeline.py --video "../19201080_before.mp4" --threshold 10
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def compute_mad_timeline(video_path, max_sec=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Vidéo : {video_path} | {fps:.1f} fps | {n_frames} frames | {W}×{H}")

    max_frames = int(max_sec * fps) if max_sec is not None else n_frames

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Impossible de lire la première frame.")

    prev_f = prev.astype(np.float32)
    mad_values = []
    timestamps = []
    frame_idx  = 1

    while True:
        ret, curr = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        curr_f = curr.astype(np.float32)
        mad = float(np.mean(np.abs(curr_f - prev_f)))
        mad_values.append(mad)
        timestamps.append(frame_idx / fps)

        prev_f = curr_f
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{min(n_frames, max_frames)} frames...", end="\r")

    cap.release()
    print(f"\n{frame_idx - 1} transitions analysées.")
    return np.array(timestamps), np.array(mad_values), fps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",     type=str,   default=os.path.join(ROOT, "Acedezinzin.mp4"))
    p.add_argument("--threshold", type=float, default=10.0,
                   help="MAD seuil au-delà duquel le modèle produit du flou (en [0-255])")
    p.add_argument("--max_sec",   type=float, default=None)
    p.add_argument("--smooth_w",  type=int,   default=15, help="Fenêtre de lissage en frames")
    p.add_argument("--out_dir",   type=str,   default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def fmt_time(x, _):
    m = int(x) // 60
    s = int(x) % 60
    return f"{m}:{s:02d}"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    timestamps, mad, fps = compute_mad_timeline(args.video, max_sec=args.max_sec)
    threshold = args.threshold

    # Lissage
    kernel  = np.ones(args.smooth_w) / args.smooth_w
    mad_smooth = np.convolve(mad, kernel, mode="same")

    # Stats
    print(f"\nMAD brut  : min={mad.min():.2f} | médiane={np.median(mad):.2f} | "
          f"90e pct={np.percentile(mad,90):.2f} | max={mad.max():.2f}")
    pct_above = 100.0 * (mad > threshold).mean()
    print(f"Frames au-delà du seuil {threshold} : {pct_above:.1f}%")

    # =========================================================================
    # Figure 1 — Courbe MAD(t) avec seuil et zones difficiles
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15, 4.5))

    # Zones difficiles (au-dessus du seuil) → fond rouge semi-transparent
    above = mad_smooth > threshold
    in_zone = False
    zone_start = None
    for i, (t, a) in enumerate(zip(timestamps, above)):
        if a and not in_zone:
            zone_start = t
            in_zone = True
        elif not a and in_zone:
            ax.axvspan(zone_start, t, color="#e74c3c", alpha=0.15, zorder=0)
            in_zone = False
    if in_zone:
        ax.axvspan(zone_start, timestamps[-1], color="#e74c3c", alpha=0.15, zorder=0)

    # Courbe MAD brute
    ax.plot(timestamps, mad, color="#95a5a6", linewidth=0.5, alpha=0.6, label="MAD brut")

    # Courbe lissée
    ax.plot(timestamps, mad_smooth, color="#2c3e50", linewidth=1.5,
            label=f"MAD lissé (fenêtre {args.smooth_w} frames)")

    # Seuil
    ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=2.0,
               label=f"Seuil = {threshold} — flou probable au-delà")

    # Annotations
    ax.set_xlabel("Temps", fontsize=11)
    ax.set_ylabel("MAD (0–255)", fontsize=11)
    title_video = os.path.basename(args.video)
    ax.set_title(
        f"MAD inter-frames — {title_video}\n"
        f"Zones rouges : MAD > {threshold} → modèle tend à produire une frame floue ({pct_above:.0f}% des transitions)",
        fontsize=11, fontweight="bold"
    )

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color="#95a5a6", linewidth=0.8, label="MAD brut"),
        plt.Line2D([0], [0], color="#2c3e50", linewidth=2, label=f"Lissé (w={args.smooth_w}f)"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2, linestyle="--",
                   label=f"Seuil = {threshold}"),
        mpatches.Patch(color="#e74c3c", alpha=0.3, label="Zone difficile (flou probable)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "mad_timeline.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out1}")

    # =========================================================================
    # Figure 2 — Histogramme MAD + seuil (pour justifier le choix du threshold)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Distribution du MAD et choix du seuil", fontsize=12, fontweight="bold")

    ax_h = axes2[0]
    ax_h.hist(mad, bins=60, color="#3498db", edgecolor="white", linewidth=0.4, alpha=0.8)
    ax_h.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=2,
                 label=f"Seuil = {threshold}")
    ax_h.axvline(np.median(mad), color="#27ae60", linestyle=":", linewidth=1.5,
                 label=f"Médiane = {np.median(mad):.1f}")
    ax_h.set_xlabel("MAD (0–255)", fontsize=10)
    ax_h.set_ylabel("Nb de transitions", fontsize=10)
    ax_h.set_title("Histogramme du MAD", fontsize=10)
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.3)

    # CDF (pour voir quel % de frames est au-dessus)
    ax_c = axes2[1]
    sorted_mad = np.sort(mad)
    cdf = np.linspace(0, 100, len(sorted_mad))
    ax_c.plot(sorted_mad, cdf, color="#2c3e50", linewidth=2)
    # Ligne verticale au seuil
    idx_thresh = np.searchsorted(sorted_mad, threshold)
    pct_below = 100 * idx_thresh / len(sorted_mad)
    ax_c.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=2,
                 label=f"Seuil = {threshold} ({100-pct_below:.0f}% au-dessus)")
    ax_c.axhline(pct_below, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.6)
    ax_c.scatter([threshold], [pct_below], color="#e74c3c", s=80, zorder=5)
    ax_c.set_xlabel("MAD (0–255)", fontsize=10)
    ax_c.set_ylabel("% de transitions ≤ MAD", fontsize=10)
    ax_c.set_title("CDF du MAD", fontsize=10)
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim(0, 100)

    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "mad_distribution.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out2}")

    # =========================================================================
    # Figure 3 — Version poster (simple, lisible, grand format)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(16, 3.8))

    # Fond zones difficiles
    in_zone = False
    for i, (t, a) in enumerate(zip(timestamps, above)):
        if a and not in_zone:
            zone_start = t
            in_zone = True
        elif not a and in_zone:
            ax3.axvspan(zone_start, t, color="#e74c3c", alpha=0.18, zorder=0, label="_")
            in_zone = False
    if in_zone:
        ax3.axvspan(zone_start, timestamps[-1], color="#e74c3c", alpha=0.18, zorder=0)

    ax3.fill_between(timestamps, 0, mad_smooth,
                     where=(mad_smooth <= threshold),
                     color="#27ae60", alpha=0.25, label="MAD ≤ seuil (OK)")
    ax3.fill_between(timestamps, 0, mad_smooth,
                     where=(mad_smooth > threshold),
                     color="#e74c3c", alpha=0.4, label=f"MAD > {threshold} (flou probable)")
    ax3.plot(timestamps, mad_smooth, color="#2c3e50", linewidth=1.8, label="MAD lissé")
    ax3.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=2.5,
                label=f"Seuil = {threshold}")

    ax3.set_xlabel("Temps", fontsize=11)
    ax3.set_ylabel("MAD inter-frames\n(0–255)", fontsize=11)
    ax3.set_title(
        f"Mouvement inter-frames sur {title_video} — zones rouges = prédiction floue attendue",
        fontsize=12, fontweight="bold"
    )
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
    ax3.set_xlim(timestamps[0], timestamps[-1])
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "mad_timeline_poster.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out3}")


if __name__ == "__main__":
    main()
