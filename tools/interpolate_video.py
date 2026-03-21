"""
tools/interpolate_video.py — Double le framerate d'une vidéo avec tiling 256x256

Supporte n'importe quelle résolution (1920x1080, 1280x714, etc.).
Tiling + overlap averaging pour éviter les artefacts de bord.

Usage:
    python tools/interpolate_video.py --input video.mp4 --output video_2x.mp4
    python tools/interpolate_video.py --input video.mp4 --output video_2x.mp4 --model gan
    python tools/interpolate_video.py --input video.mp4 --output video_2x.mp4 --model hybrid --mad_threshold 15
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_unet(ckpt_path, key="model", device="cuda"):
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


def np_to_tensor(patch_np, device):
    t = torch.from_numpy(patch_np).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def tensor_to_np(t):
    t = t.squeeze(0).cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()


@torch.no_grad()
def interpolate_tiled(model, I0_np, I1_np, device):
    H, W = I0_np.shape[:2]
    coords = get_patch_coords(H, W)
    acc   = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)
    for (y0, x0) in coords:
        p0 = np_to_tensor(I0_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE], device)
        p1 = np_to_tensor(I1_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE], device)
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
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Double le framerate d'une vidéo via UNetPixelV2 + tiling")
    p.add_argument("--input",          type=str, required=True, help="Vidéo d'entrée")
    p.add_argument("--output",         type=str, required=True, help="Vidéo de sortie (2x fps)")
    p.add_argument("--model",          type=str, default="pixel", choices=["pixel", "gan", "hybrid"],
                   help="pixel=UNetPixelV2 seul, gan=GAN seul, hybrid=pixel+GAN selon MAD")
    p.add_argument("--pixel_ckpt",     type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--gan_ckpt",       type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2_gan/best.pth"))
    p.add_argument("--mad_threshold",  type=float, default=15.0, help="Seuil MAD pour mode hybrid")
    p.add_argument("--max_frames",     type=int, default=None, help="Limite le nombre de frames")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.model}")

    # Load models
    pixel_model, ep, psnr = load_unet(args.pixel_ckpt, key="model", device=device)
    print(f"  UNetPixelV2 (pixel): epoch={ep} | psnr={psnr:.3f}")

    gan_model = None
    if args.model in ("gan", "hybrid"):
        gan_model, ep, psnr = load_unet(args.gan_ckpt, key="G", device=device)
        print(f"  UNetPixelV2 (GAN): epoch={ep} | psnr={psnr:.3f}")

    # Load video
    frames, fps, W, H = load_frames(args.input)
    if args.max_frames:
        frames = frames[:args.max_frames]
    n = len(frames)
    out_fps = fps * 2

    # Interpolate
    output = []
    gan_count = 0
    t0 = time.time()
    print(f"\nInterpolating {n-1} pairs (tiling {PATCH_SIZE}px, overlap={OVERLAP}px)...")

    for i in range(n - 1):
        I0, I1 = frames[i], frames[i + 1]

        if args.model == "pixel":
            Imid = interpolate_tiled(pixel_model, I0, I1, device)
        elif args.model == "gan":
            Imid = interpolate_tiled(gan_model, I0, I1, device)
        else:  # hybrid
            mad = compute_mad(I0, I1)
            if mad > args.mad_threshold:
                Imid = interpolate_tiled(gan_model, I0, I1, device)
                gan_count += 1
            else:
                Imid = interpolate_tiled(pixel_model, I0, I1, device)

        output.append(I0)
        output.append(Imid)

        if (i + 1) % 20 == 0 or i == n - 2:
            elapsed = time.time() - t0
            extra = f" | GAN: {gan_count}/{i+1}" if args.model == "hybrid" else ""
            print(f"  {i+1}/{n-1} | {elapsed:.1f}s{extra}")

    output.append(frames[-1])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video(output, args.output, out_fps, W, H)
    print(f"\nDone. {fps}fps -> {out_fps}fps | {n} -> {len(output)} frames")


if __name__ == "__main__":
    main()
