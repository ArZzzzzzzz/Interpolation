"""
poster/18_interpolate_4x.py — Applique une 2ème passe d'interpolation sur apple_pixel.mp4
48fps → 96fps
"""
import os, sys, time
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2

VIDEO_IN   = os.path.join(os.path.dirname(__file__), "results/apple_interp/apple_pixel.mp4")
VIDEO_OUT  = os.path.join(os.path.dirname(__file__), "results/apple_interp/apple_pixel_4x.mp4")
PIXEL_CKPT = os.path.join(ROOT, "checkpoints/pixel_v2/best.pth")

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_unet(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    m = UNetPixelV2(
        base_ch      = args_.get("base_ch", 64),
        use_residual = not args_.get("no_residual", False),
        use_refine   = not args_.get("no_refine", False),
    ).to(device)
    m.load_state_dict(ckpt.get("model"))
    m.eval()
    return m


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
def interpolate_tiled(model, I0_np, I1_np):
    H, W   = I0_np.shape[:2]
    coords = get_patch_coords(H, W)
    acc    = np.zeros((H, W, 3), dtype=np.float32)
    count  = np.zeros((H, W, 1), dtype=np.float32)
    for (y0, x0) in coords:
        p0   = np_to_tensor(I0_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        p1   = np_to_tensor(I1_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE])
        pred = tensor_to_np(model(p0, p1)).astype(np.float32)
        acc  [y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += pred
        count[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0
    return (acc / count).clip(0, 255).astype(np.uint8)


# Load video
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
print(f"Loaded {len(frames)} frames | {fps}fps | {W}x{H}")

model = load_unet(PIXEL_CKPT)

# Interpolate
out_frames = []
n = len(frames)
t0 = time.time()
print(f"Interpolating {n-1} pairs...")

for i in range(n - 1):
    Imid = interpolate_tiled(model, frames[i], frames[i+1])
    out_frames.append(frames[i])
    out_frames.append(Imid)
    if (i + 1) % 50 == 0 or i == n - 2:
        print(f"  {i+1}/{n-1} | {time.time()-t0:.1f}s")

out_frames.append(frames[-1])

# Save
out_fps = fps * 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, out_fps, (W, H))
for f in out_frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print(f"Saved: {VIDEO_OUT}  ({len(out_frames)} frames @ {out_fps}fps)")
