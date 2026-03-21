"""
poster/17_interpolate_apple.py — Interpole la vidéo APPLE avec tiling 256x256

Génère 2 vidéos dans poster/results/apple_interp/ :
  - apple_pixel.mp4   : UNetPixelV2 pur
  - apple_hybrid.mp4  : UNetPixelV2 (faible mouvement) + GAN (fort mouvement, MAD > seuil)

Tiling avec overlap=32px pour gérer la résolution 1280x714.
"""
import os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2

VIDEO_IN   = os.path.join(os.path.dirname(__file__), "APPLE VIDEO Satisfying Apple Cutting ASMR Relaxing Knife Sounds.mp4")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results", "apple_interp")
PIXEL_CKPT = os.path.join(ROOT, "checkpoints/pixel_v2/best.pth")
GAN_CKPT   = os.path.join(ROOT, "checkpoints/pixel_v2_gan/best.pth")

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP
MAD_THRESH = 15.0   # seuil hybride (sur [0,255])

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_unet(ckpt_path, key="model"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    m = UNetPixelV2(
        base_ch      = args_.get("base_ch", 64),
        use_residual = not args_.get("no_residual", False),
        use_refine   = not args_.get("no_refine", False),
    ).to(device)
    state = ckpt.get(key, ckpt.get("model"))
    m.load_state_dict(state)
    m.eval()
    return m, ckpt.get("epoch", "?"), ckpt.get("best_psnr", 0)


def get_patch_coords(H, W):
    coords = []
    y = 0
    while True:
        y0 = min(y, H - PATCH_SIZE)
        x = 0
        while True:
            x0 = min(x, W - PATCH_SIZE)
            coords.append((y0, x0))
            if x0 + PATCH_SIZE >= W:
                break
            x += STRIDE
        if y0 + PATCH_SIZE >= H:
            break
        y += STRIDE
    return coords


def np_to_tensor(patch_np):
    """[H,W,3] uint8 → [1,3,H,W] float [-1,1]"""
    t = torch.from_numpy(patch_np).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def tensor_to_np(t):
    """[1,3,H,W] float [-1,1] → [H,W,3] uint8"""
    t = t.squeeze(0).cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()


@torch.no_grad()
def interpolate_tiled(model, I0_np, I1_np):
    """Interpolation avec tiling + overlap averaging."""
    H, W = I0_np.shape[:2]
    coords = get_patch_coords(H, W)

    acc   = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    for (y0, x0) in coords:
        p0 = np_to_tensor(I0_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        p1 = np_to_tensor(I1_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        pred = model(p0, p1)
        pred_np = tensor_to_np(pred).astype(np.float32)
        acc  [y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += pred_np
        count[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0

    return (acc / count).clip(0, 255).astype(np.uint8)


def compute_mad(I0_np, I1_np):
    return float(np.mean(np.abs(I0_np.astype(np.float32) - I1_np.astype(np.float32))))


def load_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames | {fps}fps | {W}x{H}")
    return frames, fps, W, H


def save_video(frames, path, fps, W, H):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved: {path}  ({len(frames)} frames @ {fps}fps)")


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

print(f"Loading UNetPixelV2 (pixel)...")
pixel_model, ep, psnr = load_unet(PIXEL_CKPT, key="model")
print(f"  epoch={ep} | best_psnr={psnr:.3f}")

print(f"Loading UNetPixelV2 (GAN)...")
gan_model, ep, psnr = load_unet(GAN_CKPT, key="G")
print(f"  epoch={ep} | best_psnr={psnr:.3f}")

# ---------------------------------------------------------------------------
# Load video
# ---------------------------------------------------------------------------

frames, fps, W, H = load_frames(VIDEO_IN)
n = len(frames)
out_fps = fps * 2

# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

pixel_out  = []
hybrid_out = []
gan_count  = 0

print(f"\nInterpolating {n-1} pairs (tiling {PATCH_SIZE}px, overlap={OVERLAP}px)...")
t0 = time.time()

for i in range(n - 1):
    I0 = frames[i]
    I1 = frames[i + 1]

    mad = compute_mad(I0, I1)

    # --- Pixel ---
    Imid_pixel = interpolate_tiled(pixel_model, I0, I1)
    pixel_out.append(I0)
    pixel_out.append(Imid_pixel)

    # --- Hybrid ---
    if mad > MAD_THRESH:
        Imid_hybrid = interpolate_tiled(gan_model, I0, I1)
        gan_count += 1
    else:
        Imid_hybrid = Imid_pixel  # réutilise pixel déjà calculé

    hybrid_out.append(I0)
    hybrid_out.append(Imid_hybrid)

    if (i + 1) % 20 == 0 or i == n - 2:
        elapsed = time.time() - t0
        print(f"  {i+1}/{n-1} | MAD={mad:.1f} | GAN used: {gan_count}/{i+1} | {elapsed:.1f}s")

pixel_out.append(frames[-1])
hybrid_out.append(frames[-1])

print(f"\nGAN applied to {gan_count}/{n-1} pairs ({100*gan_count/(n-1):.1f}%, MAD>{MAD_THRESH})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

save_video(pixel_out,  os.path.join(OUT_DIR, "apple_pixel.mp4"),  out_fps, W, H)
save_video(hybrid_out, os.path.join(OUT_DIR, "apple_hybrid.mp4"), out_fps, W, H)
print(f"\nDone. Output in {OUT_DIR}/")
