"""
motion_analysis.py — Analyse du mouvement inter-frame d'une vidéo.

Métriques calculées :
  - Optical Flow Farneback : magnitude moyenne du flow (pixels/frame) — recommandée
  - Frame difference (MAD) : différence absolue moyenne (0-255) — simple

Usage:
    python motion_analysis.py --video Acedezinzin.mp4
    python motion_analysis.py --video Acedezinzin.mp4 --fps_out 25
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_motion(video_path: str, max_sec: float = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vidéo : {video_path}")
    print(f"FPS : {fps:.2f} | Frames : {n_frames}")

    max_frames = int(max_sec * fps) if max_sec is not None else n_frames
    print(f"Analyse limitée à {max_sec}s ({max_frames} frames)" if max_sec else "Analyse complète")

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Impossible de lire la première frame.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_float = prev.astype(np.float32)

    flow_magnitudes = []
    mad_values      = []
    timestamps      = []
    frame_idx       = 1

    while True:
        ret, curr = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        curr_gray  = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_float = curr.astype(np.float32)

        # --- Optical Flow Farneback ---
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(float(magnitude.mean()))

        # --- Frame difference (MAD) ---
        mad = float(np.mean(np.abs(curr_float - prev_float)))
        mad_values.append(mad)

        timestamps.append(frame_idx / fps)
        frame_idx += 1

        prev_gray  = curr_gray
        prev_float = curr_float

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{n_frames} frames traitées...", end="\r")

    cap.release()
    print(f"\n{frame_idx - 1} transitions analysées.")

    return np.array(timestamps), np.array(flow_magnitudes), np.array(mad_values), fps


def plot_motion(timestamps, flow_mag, mad, fps, out_path="motion_analysis.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Analyse du mouvement inter-frame", fontsize=14, fontweight="bold")

    # --- Optical Flow ---
    ax = axes[0]
    ax.plot(timestamps, flow_mag, color="steelblue", linewidth=0.8, alpha=0.9, label="Flow moyen")
    # Lissage
    if len(flow_mag) > 20:
        kernel = np.ones(15) / 15
        smooth = np.convolve(flow_mag, kernel, mode="same")
        ax.plot(timestamps, smooth, color="red", linewidth=1.5, label="Lissé (fenêtre 15f)")
    # Stats
    p50 = np.percentile(flow_mag, 50)
    p75 = np.percentile(flow_mag, 75)
    ax.axhline(p50, color="green",  linestyle="--", linewidth=1.0, label=f"Médiane : {p50:.2f} px/f")
    ax.axhline(p75, color="orange", linestyle="--", linewidth=1.0, label=f"75e pct : {p75:.2f} px/f")
    ax.set_ylabel("Flow magnitude (pixels/frame)")
    ax.set_title("Optical Flow Farneback — métrique recommandée")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- MAD ---
    ax2 = axes[1]
    ax2.plot(timestamps, mad, color="purple", linewidth=0.8, alpha=0.9, label="MAD")
    if len(mad) > 20:
        smooth_mad = np.convolve(mad, np.ones(15) / 15, mode="same")
        ax2.plot(timestamps, smooth_mad, color="red", linewidth=1.5, label="Lissé")
    ax2.axhline(np.percentile(mad, 50), color="green",  linestyle="--", linewidth=1.0,
                label=f"Médiane : {np.percentile(mad,50):.2f}")
    ax2.axhline(np.percentile(mad, 75), color="orange", linestyle="--", linewidth=1.0,
                label=f"75e pct : {np.percentile(mad,75):.2f}")
    ax2.set_ylabel("MAD (0–255)")
    ax2.set_xlabel("Temps (secondes)")
    ax2.set_title("Frame Difference (MAD) — simple mais moins précis")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x//60)}:{int(x%60):02d}"
    ))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Graphique sauvegardé : {out_path}")

    # Stats résumé
    print(f"\n--- Optical Flow (pixels/frame) ---")
    print(f"  Min    : {flow_mag.min():.3f}")
    print(f"  Médiane: {np.median(flow_mag):.3f}")
    print(f"  75e pct: {np.percentile(flow_mag, 75):.3f}")
    print(f"  90e pct: {np.percentile(flow_mag, 90):.3f}")
    print(f"  Max    : {flow_mag.max():.3f}")
    print(f"\nSuggestion de seuil : entre médiane ({np.median(flow_mag):.2f}) "
          f"et 75e pct ({np.percentile(flow_mag, 75):.2f}) px/frame")

    # Lissage temporel (fenetre 9 frames) avant detection
    kernel = np.ones(9) / 9
    flow_smooth = np.convolve(flow_mag, kernel, mode="same")

    # Segments au-dessus du seuil (90e pct sur signal lisse)
    threshold = np.percentile(flow_smooth, 90)
    above = flow_smooth > threshold
    segments = []
    in_seg = False
    for idx, val in enumerate(above):
        t = timestamps[idx]
        if val and not in_seg:
            seg_start = t
            in_seg = True
        elif not val and in_seg:
            segments.append((seg_start, t))
            in_seg = False
    if in_seg:
        segments.append((seg_start, timestamps[-1]))

    print(f"\nSegments au-dessus du seuil ({threshold:.2f} px/frame, lisse, 90e pct) :")
    for s, e in segments:
        print(f"  {s:.2f}s -> {e:.2f}s  (duree {e-s:.2f}s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",    type=str,   default="Acedezinzin.mp4")
    p.add_argument("--out",      type=str,   default="motion_analysis.png")
    p.add_argument("--max_sec",  type=float, default=None, help="Limiter l'analyse aux N premières secondes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamps, flow_mag, mad, fps = compute_motion(args.video, max_sec=args.max_sec)
    plot_motion(timestamps, flow_mag, mad, fps, out_path=args.out)
