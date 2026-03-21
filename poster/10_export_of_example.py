"""
poster/10_export_of_example.py — Exporte I0, I1, Imid_OF pour le poster

Usage:
    cd poster
    python 10_export_of_example.py --data_root "D:/projet mumltimodal/vimeo_triplet"
    python 10_export_of_example.py --idx 42   # choisir un triplet précis
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders


def tensor_to_pil(t):
    """[1,3,H,W] in [-1,1] → PIL RGB."""
    t = t.squeeze(0).detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255.0).byte()
    return Image.fromarray(t.permute(1, 2, 0).numpy(), "RGB")


def compute_farneback_flow(I0_np255, I1_np255):
    g0 = cv2.cvtColor(I0_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(I1_np255.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_tensor(img_t, flow_np, t=0.5):
    H, W = img_t.shape[-2:]
    flow = torch.from_numpy(flow_np).float()
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    grid = torch.stack([gx + t * flow[..., 0] / (W / 2),
                        gy + t * flow[..., 1] / (H / 2)], dim=-1).unsqueeze(0)
    return F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def tensor_to_np255(t):
    t = t.detach().cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--idx",       type=int, default=0, help="Index du triplet dans le val set")
    p.add_argument("--out_dir",   type=str, default=os.path.join(os.path.dirname(__file__), "results/of_example"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _, val_loader, _ = get_dataloaders(args.data_root, crop_size=256, batch_size=1, num_workers=0)

    for i, batch in enumerate(val_loader):
        if i < args.idx:
            continue

        I0   = batch["I0"]    # [1,3,H,W]
        Imid = batch["Imid"]
        I1   = batch["I1"]

        I0_np = tensor_to_np255(I0)[0]
        I1_np = tensor_to_np255(I1)[0]

        flow     = compute_farneback_flow(I0_np, I1_np)
        w0       = warp_tensor(I0, flow,  t= 0.5)
        w1       = warp_tensor(I1, flow,  t=-0.5)
        Imid_of  = (0.5 * w0 + 0.5 * w1).clamp(-1, 1)

        tensor_to_pil(I0).save(os.path.join(args.out_dir, "I0.png"))
        tensor_to_pil(I1).save(os.path.join(args.out_dir, "I1.png"))
        tensor_to_pil(Imid).save(os.path.join(args.out_dir, "Imid_GT.png"))
        tensor_to_pil(Imid_of).save(os.path.join(args.out_dir, "Imid_OF.png"))

        print(f"Triplet #{args.idx} sauvegarde dans {args.out_dir}/")
        print("  I0.png  |  I1.png  |  Imid_GT.png  |  Imid_OF.png")
        break


if __name__ == "__main__":
    main()
