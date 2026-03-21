"""
poster/19_interpolate_hybrid_diff.py — Hybride UNetPixelV2 + DiffusionV3 warm start

MAD <= MAD_THRESH : UNetPixelV2 (rapide)
MAD >  MAD_THRESH : DiffusionV3 warm start depuis Imid_pixel (qualité)

Résultat : poster/results/apple_interp/apple_hybrid_diff.mp4
"""
import os, sys, time
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2
from models.unet_latent   import FrozenVAE
from models.diffusion_v3  import DiffusionUNetV3, CosineNoiseSchedule

VIDEO_IN   = os.path.join(os.path.dirname(__file__), "APPLE VIDEO Satisfying Apple Cutting ASMR Relaxing Knife Sounds.mp4")
VIDEO_OUT  = os.path.join(os.path.dirname(__file__), "results/apple_interp/apple_hybrid_diff.mp4")
PIXEL_CKPT = os.path.join(ROOT, "checkpoints/pixel_v2/best.pth")
DIFF_CKPT  = os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth")

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP
MAD_THRESH = 3.0    # seuil adapté au MAD réel de cette vidéo (~6.5 max)
T          = 1000
T_START    = 200    # niveau de bruit warm start
T_INF      = 20     # steps DDIM (réduit pour la vitesse)
BASE_CH    = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | MAD_THRESH={MAD_THRESH} | T_START={T_START} | T_INF={T_INF}")


# ---------------------------------------------------------------------------
# Helpers tiling
# ---------------------------------------------------------------------------

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


def np_to_tensor(p):
    return (torch.from_numpy(p).float().permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(device)

def tensor_to_np(t):
    t = t.squeeze(0).cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()


@torch.no_grad()
def interp_pixel_tiled(model, I0_np, I1_np):
    H, W   = I0_np.shape[:2]
    acc    = np.zeros((H, W, 3), dtype=np.float32)
    count  = np.zeros((H, W, 1), dtype=np.float32)
    for (y0, x0) in get_patch_coords(H, W):
        p0   = np_to_tensor(I0_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        p1   = np_to_tensor(I1_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        pred = tensor_to_np(model(p0, p1)).astype(np.float32)
        acc  [y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += pred
        count[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0
    return (acc / count).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def interp_diffusion_tiled(pixel_model, vae, diff_model, schedule, I0_np, I1_np):
    """Warm start patch par patch : pixel → bruiter → DDIM → décoder."""
    H, W   = I0_np.shape[:2]
    acc    = np.zeros((H, W, 3), dtype=np.float32)
    count  = np.zeros((H, W, 1), dtype=np.float32)
    ab     = schedule["alpha_bar"]

    for (y0, x0) in get_patch_coords(H, W):
        p0 = np_to_tensor(I0_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        p1 = np_to_tensor(I1_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])

        # 1. Pixel prediction (coarse)
        Imid_px = pixel_model(p0, p1).clamp(-1, 1)

        # 2. Encode tout en latent
        z0      = vae.encode(p0)
        z1      = vae.encode(p1)
        z_pixel = vae.encode(Imid_px)

        # 3. Warm start : bruiter z_pixel à T_START
        t_tensor = torch.tensor([T_START], device=device, dtype=torch.long)
        z_noisy, _ = schedule.q_sample(z_pixel, t_tensor)

        # 4. DDIM T_START → 0
        z = z_noisy.clone()
        timesteps = torch.linspace(T_START, 1, T_INF, dtype=torch.long).to(device)
        ts_prev   = torch.cat([timesteps[1:], torch.zeros(1, dtype=torch.long, device=device)])

        for t_cur, t_prev in zip(timesteps, ts_prev):
            t_batch = torch.full((1,), t_cur.item(), device=device, dtype=torch.long)
            out     = diff_model(z, t_batch, z0, z1)
            a_cur   = ab[t_cur].to(device)
            a_prev  = ab[t_prev].to(device) if t_prev > 0 else torch.tensor(1.0, device=device)
            x0_pred = (a_cur.sqrt() * z - (1 - a_cur).sqrt() * out).clamp(-10, 10)
            eps_eq  = (1 - a_cur).sqrt() * z + a_cur.sqrt() * out
            z       = a_prev.sqrt() * x0_pred + (1 - a_prev).clamp(0).sqrt() * eps_eq

        # 5. Decode
        pred = tensor_to_np(vae.decode(z)).astype(np.float32)
        acc  [y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += pred
        count[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0

    return (acc / count).clip(0, 255).astype(np.uint8)


def compute_mad(I0, I1):
    return float(np.mean(np.abs(I0.astype(np.float32) - I1.astype(np.float32))))


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

print("Loading UNetPixelV2...")
px_ckpt = torch.load(PIXEL_CKPT, map_location=device, weights_only=False)
px_args = px_ckpt.get("args", {})
pixel_model = UNetPixelV2(
    base_ch      = px_args.get("base_ch", 64),
    use_residual = not px_args.get("no_residual", False),
    use_refine   = not px_args.get("no_refine", False),
).to(device)
pixel_model.load_state_dict(px_ckpt["model"])
pixel_model.eval()
print(f"  epoch={px_ckpt.get('epoch','?')} | psnr={px_ckpt.get('best_psnr',0):.3f}")

print("Loading VAE...")
vae = FrozenVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)

print("Loading DiffusionUNetV3...")
d_ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=False)
diff_model = DiffusionUNetV3(latent_ch=4, base_ch=BASE_CH).to(device)
diff_model.load_state_dict(d_ckpt["model"])
diff_model.eval()
schedule = CosineNoiseSchedule(T=T).to(device)
print(f"  epoch={d_ckpt.get('epoch','?')} | psnr={d_ckpt.get('best_psnr',0):.3f}")

# ---------------------------------------------------------------------------
# Load video
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_IN)
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
n = len(frames)
print(f"Loaded {n} frames | {fps}fps | {W}x{H}")

# ---------------------------------------------------------------------------
# Interpolation hybride
# ---------------------------------------------------------------------------

out_frames = []
diff_count = 0
t0 = time.time()
print(f"\nInterpolating {n-1} pairs (hybrid pixel/diffusion)...")

for i in range(n - 1):
    I0  = frames[i]
    I1  = frames[i + 1]
    mad = compute_mad(I0, I1)

    if mad > MAD_THRESH:
        Imid = interp_diffusion_tiled(pixel_model, vae, diff_model, schedule, I0, I1)
        diff_count += 1
    else:
        Imid = interp_pixel_tiled(pixel_model, I0, I1)

    out_frames.append(I0)
    out_frames.append(Imid)

    if (i + 1) % 10 == 0 or i == n - 2:
        elapsed = time.time() - t0
        print(f"  {i+1}/{n-1} | MAD={mad:.1f} | diff used: {diff_count}/{i+1} | {elapsed:.1f}s")

out_frames.append(frames[-1])

print(f"\nDiffusion applied to {diff_count}/{n-1} pairs ({100*diff_count/(n-1):.1f}%)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps * 2, (W, H))
for f in out_frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print(f"Saved: {VIDEO_OUT}  ({len(out_frames)} frames @ {fps*2}fps)")
