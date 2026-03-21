"""
eval_diffusion_v3.py — Génère des GIFs depuis bruit pur avec DiffusionUNetV3

Pour chaque sample : I0 → Imid_pred → I1 en boucle (depuis bruit pur complet)

Usage:
    python eval_diffusion_v3.py --ckpt checkpoints/diffusion_v3/epoch_035.pth --n 10
"""

import argparse
import os
import torch
from PIL import Image
from torch.amp import autocast

from dataset import get_dataloaders
from models.unet_latent import FrozenVAE
from models.diffusion_v3 import DiffusionUNetV3, CosineNoiseSchedule, DDIMSampler


def to_pil(t):
    t = t.cpu().float().clamp(-1, 1)
    return Image.fromarray(((t + 1.0) / 2.0 * 255).permute(1, 2, 0).byte().numpy())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str,   required=True)
    p.add_argument("--data_root", type=str,   default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--out_dir",   type=str,   default=None)
    p.add_argument("--n",         type=int,   default=10)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--T_inf",     type=int,   default=250)
    p.add_argument("--base_ch",   type=int,   default=128)
    p.add_argument("--vae_id",    type=str,   default="stabilityai/sd-vae-ft-mse")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.ckpt), "eval_gifs"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading VAE...")
    vae = FrozenVAE(model_id=args.vae_id).to(device)

    print(f"Loading model from {args.ckpt}...")
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiffusionUNetV3(latent_ch=4, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']} | best_psnr={ckpt.get('best_psnr', 0):.3f}")

    schedule = CosineNoiseSchedule(T=args.T).to(device)
    sampler  = DDIMSampler(model, schedule, T_inf=args.T_inf, eta=0.0, prediction_type="v")

    _, val_loader, _ = get_dataloaders(
        args.data_root, crop_size=256, batch_size=1, num_workers=2
    )

    print(f"\nGenerating {args.n} GIFs (T_inf={args.T_inf} steps from pure noise)...\n")

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= args.n:
                break

            I0   = batch["I0"].to(device)
            Imid = batch["Imid"].to(device)
            I1   = batch["I1"].to(device)

            z0 = vae.encode(I0)
            z1 = vae.encode(I1)

            # Depuis bruit pur
            z_pred     = sampler.sample(z0, z1, device=device)
            pred_pixel = vae.decode(z_pred)

            pil_i0   = to_pil(I0.squeeze(0))
            pil_pred = to_pil(pred_pixel.squeeze(0))
            pil_gt   = to_pil(Imid.squeeze(0))
            pil_i1   = to_pil(I1.squeeze(0))

            # GIF : I0 → pred → GT → I1
            gif_path = os.path.join(out_dir, f"sample_{count:03d}.gif")
            pil_i0.save(
                gif_path, save_all=True,
                append_images=[pil_pred, pil_gt, pil_i1],
                duration=500, loop=0,
            )

            print(f"  [{count+1:02d}/{args.n}] {gif_path}")
            count += 1

    print(f"\nDone. GIFs in {out_dir}/")
    print("GIF order: I0 -> pred -> GT -> I1")


if __name__ == "__main__":
    main()
