"""
tools/generate_gifs.py — Génère des GIFs de comparaison depuis le dataset Vimeo-90K val

Pour chaque sample, génère 2 GIFs :
  - sample_XXX_gt.gif   : I0 → Imid_GT → I1 (boucle)
  - sample_XXX_pred.gif : I0 → Imid_pred → I1 (boucle)

Usage:
    python tools/generate_gifs.py --n 10
    python tools/generate_gifs.py --n 5 --out_dir my_gifs
    python tools/generate_gifs.py --n 10 --ckpt checkpoints/pixel_v2_gan/best.pth --ckpt_key G
"""
import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2


def tensor_to_pil(t):
    t = t.cpu().clamp(-1, 1)
    return TF.to_pil_image((t + 1.0) / 2.0)


def add_label(img, text):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    draw.text((2, 2), text, fill=(255, 255, 255))
    return img


def make_gif(frames, path, duration=400):
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=duration)


def parse_args():
    p = argparse.ArgumentParser(description="Génère des GIFs GT vs prediction depuis Vimeo-90K val")
    p.add_argument("--data_root", type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--ckpt",      type=str, default=os.path.join(ROOT, "checkpoints/pixel_v2/best.pth"))
    p.add_argument("--ckpt_key",  type=str, default="model", help="Clé du state_dict dans le checkpoint (model ou G)")
    p.add_argument("--out_dir",   type=str, default=os.path.join(ROOT, "tools", "gifs"))
    p.add_argument("--n",         type=int, default=10, help="Nombre de GIFs à générer")
    p.add_argument("--duration",  type=int, default=400, help="Durée par frame en ms")
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | n={args.n} | output: {args.out_dir}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    args_ = ckpt.get("args", {})
    model = UNetPixelV2(
        base_ch      = args_.get("base_ch", 64),
        use_residual = not args_.get("no_residual", False),
        use_refine   = not args_.get("no_refine", False),
    ).to(device)
    model.load_state_dict(ckpt.get(args.ckpt_key, ckpt.get("model")))
    model.eval()
    print(f"Model loaded from {args.ckpt} (key={args.ckpt_key})")

    # Load val dataset
    _, val_loader, _ = get_dataloaders(args.data_root, crop_size=256, batch_size=1, num_workers=0)

    print(f"\nGenerating {args.n} GIF pairs...")
    for i, batch in enumerate(val_loader):
        if i >= args.n:
            break

        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        pred = model(I0, I1)

        pil_I0   = tensor_to_pil(I0[0])
        pil_gt   = tensor_to_pil(Imid[0])
        pil_pred = tensor_to_pil(pred[0])
        pil_I1   = tensor_to_pil(I1[0])

        # GIF GT : I0 → Imid_GT → I1
        gt_frames = [
            add_label(pil_I0, "I0"),
            add_label(pil_gt, "Imid (GT)"),
            add_label(pil_I1, "I1"),
        ]
        gt_path = os.path.join(args.out_dir, f"sample_{i:03d}_gt.gif")
        make_gif(gt_frames, gt_path, args.duration)

        # GIF Pred : I0 → Imid_pred → I1
        pred_frames = [
            add_label(pil_I0,   "I0"),
            add_label(pil_pred, "Imid (pred)"),
            add_label(pil_I1,   "I1"),
        ]
        pred_path = os.path.join(args.out_dir, f"sample_{i:03d}_pred.gif")
        make_gif(pred_frames, pred_path, args.duration)

        print(f"  [{i+1}/{args.n}] {gt_path} + {pred_path}")

    print(f"\nDone. GIFs in {args.out_dir}/")


if __name__ == "__main__":
    main()
