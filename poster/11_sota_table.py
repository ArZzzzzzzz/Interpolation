"""
poster/11_sota_table.py — Tableau état de l'art VFI pour le poster
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(__file__), "results", "sota_table.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

methods = [
    ("SuperSloMo",  "2018", "U-Net + joint\noptical flow",    33.2, False),
    ("DAIN",        "2019", "Depth-aware\noptical flow",       34.7, False),
    ("RIFE",        "2022", "Intermediate\nflow estimation",   35.6, False),
    ("FILM",        "2022", "Multi-scale\nfeature pyramid",    36.1, False),
    ("LDMVFI",      "2023", "Latent diffusion",                35.4, False),
    ("Ours\n(PixelV2)", "2024", "Direct pixel\nregression",   33.3, True),
]

COL_W  = [2.2, 1.0, 2.4, 1.8]   # Method | Year | Approach | PSNR
HEADERS = ["Method", "Year", "Approach", "PSNR ↑\n(Vimeo-90K)"]

COLOR_HEADER = "#1a1a2e"
COLOR_OUR    = "#e8f4fd"
COLOR_ALT    = "#f8f9fa"
COLOR_WHITE  = "#ffffff"
COLOR_BEST   = "#d4edda"
ACCENT       = "#2980b9"

fig_w = sum(COL_W) + 0.6
row_h = 0.72
fig_h = row_h * (len(methods) + 1) + 0.5

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, sum(COL_W))
ax.set_ylim(0, fig_h)
ax.axis("off")

def draw_cell(ax, x, y, w, h, text, bg, fg="black", fontsize=9, bold=False, va="center"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor="#dee2e6", linewidth=0.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va=va,
            fontsize=fontsize, color=fg,
            fontweight="bold" if bold else "normal",
            linespacing=1.3)

# Header
y_header = fig_h - row_h - 0.15
x = 0
for i, (header, cw) in enumerate(zip(HEADERS, COL_W)):
    draw_cell(ax, x, y_header, cw, row_h, header,
              bg=COLOR_HEADER, fg="white", fontsize=9.5, bold=True)
    x += cw

# Rows
for row_i, (name, year, approach, psnr, is_ours) in enumerate(methods):
    y = y_header - row_h * (row_i + 1)
    bg = COLOR_OUR if is_ours else (COLOR_ALT if row_i % 2 == 0 else COLOR_WHITE)

    # Best PSNR highlight
    best_psnr = max(m[3] for m in methods if not m[4])
    psnr_bg = COLOR_BEST if (psnr == best_psnr and not is_ours) else bg

    x = 0
    draw_cell(ax, x, y, COL_W[0], row_h, name,  bg=bg,       fontsize=9,  bold=is_ours); x += COL_W[0]
    draw_cell(ax, x, y, COL_W[1], row_h, year,  bg=bg,       fontsize=8.5); x += COL_W[1]
    draw_cell(ax, x, y, COL_W[2], row_h, approach, bg=bg,    fontsize=8.5); x += COL_W[2]
    draw_cell(ax, x, y, COL_W[3], row_h, f"{psnr:.1f} dB", bg=psnr_bg, fontsize=9, bold=(psnr==best_psnr or is_ours))

    # Barre latérale pour "Ours"
    if is_ours:
        ax.add_patch(plt.Rectangle((0, y), 0.06, row_h, facecolor=ACCENT, linewidth=0))

# Légende
legend_y = 0.08
handles = [
    mpatches.Patch(facecolor=COLOR_BEST,  edgecolor="gray", label="Best PSNR (prior work)"),
    mpatches.Patch(facecolor=COLOR_OUR,   edgecolor="gray", label="Our method"),
    mpatches.Patch(facecolor=ACCENT,      label=""),
]
ax.legend(handles=handles[:2], loc="lower center",
          bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7.5,
          framealpha=0.9, edgecolor="#dee2e6")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
