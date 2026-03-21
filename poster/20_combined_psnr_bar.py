"""
poster/20_combined_psnr_bar.py — Toutes les barres PSNR sur un seul graphique
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "combined_psnr_bar.png")

models = [
    ("Moyenne\n(I0+I1)/2",          24.64, "#b0bec5", "Baselines"),
    ("Optical Flow\n(Farneback)",    21.88, "#90a4ae", "Baselines"),
    ("UNetPixelV2",                  33.26, "#2980b9", "Ours"),
    ("UNetPixelV2\n+ GAN",           32.30, "#8e44ad", "Ours"),
    ("Diffusion pure\n(50 steps)",   26.50, "#e67e22", "Diffusion"),
    ("Warm-start\nt=200 (50 steps)", 27.30, "#e74c3c", "Diffusion"),
]

labels  = [m[0] for m in models]
psnrs   = [m[1] for m in models]
colors  = [m[2] for m in models]
groups  = [m[3] for m in models]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(x, psnrs, color=colors, edgecolor="white", linewidth=0.8, width=0.6, zorder=3)

# Valeurs au-dessus des barres
for bar, val in zip(bars, psnrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("PSNR (dB)", fontsize=11)
ax.set_title("PSNR comparison — all models (Vimeo-90K val)", fontsize=12, fontweight="bold")
ax.set_ylim(18, 37)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Légende groupes
legend_handles = [
    mpatches.Patch(facecolor="#b0bec5", label="Baselines"),
    mpatches.Patch(facecolor="#2980b9", label="Ours (pixel)"),
    mpatches.Patch(facecolor="#e67e22", label="Diffusion"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
