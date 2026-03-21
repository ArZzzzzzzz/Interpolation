"""
eval_warmstart.py — Test du warm start depuis z_gt bruité à t_start

Pipeline testée ici (meilleur cas possible, z_gt connu) :
    z_gt  = VAE.encode(Imid_gt)
    z_t   = q_sample(z_gt, t=t_start)       # bruit au niveau t_start
    z_0   = DDIM(z_t, t_start → 0, steps)   # dénoise conditionné sur z0, z1
    pred  = VAE.decode(z_0)

Pipeline finale (en production) :
    z_pixel = VAE.encode(UNetPixel(I0, I1))  # remplace z_gt
    z_t     = q_sample(z_pixel, t=t_start)
    z_0     = DDIM(z_t, t_start → 0, steps)
    pred    = VAE.decode(z_0)

Usage:
    python eval_warmstart.py --ckpt checkpoints/diffusion/epoch_020.pth
    python eval_warmstart.py --ckpt checkpoints/diffusion/epoch_020.pth --t_start 50 --steps 50 --n 10
"""

import argparse
import os

import torch
import numpy as np
from PIL import Image, ImageDraw

from dataset import get_dataloaders
from models.unet_latent import FrozenVAE
from models.diffusion import DiffusionUNet, CosineNoiseSchedule, DDIMSampler
from metrics import compute_psnr, compute_ssim


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).byte()
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def add_label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    draw.text((2, 2), text, fill=(255, 255, 255))
    return img


def make_gif(frames, path: str, duration: int = 500):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )


def ddim_from_t(model, schedule, z_t, t_start, steps, z0, z1, device):
    """
    DDIM de t=t_start vers t=0 en `steps` pas.
    z_t : latent bruité au niveau t_start.
    """
    ab = schedule["alpha_bar"].to(device)

    timesteps = torch.linspace(t_start, 0, steps + 1, dtype=torch.long).to(device)
    ts_cur  = timesteps[:-1]
    ts_prev = timesteps[1:]

    z = z_t.clone()
    B = z.shape[0]

    for t_cur, t_prev in zip(ts_cur, ts_prev):
        t_batch = torch.full((B,), t_cur.item(), device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(z, t_batch, z0, z1)

        a_cur  = ab[t_cur]
        a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        x0_pred = (z - (1 - a_cur).sqrt() * eps) / a_cur.sqrt().clamp(min=1e-6)
        x0_pred = x0_pred.clamp(-10, 10)
        dir_xt  = (1 - a_prev).clamp(0).sqrt() * eps
        z       = a_prev.sqrt() * x0_pred + dir_xt

    return z


def ddim_full_from_noisy(model, schedule, z_start, steps, z0, z1, device):
    """
    Schedule COMPLET (t=T-1 → 0) mais en partant de z_start (image bruitée à t_start)
    au lieu de bruit pur. Le modèle à haut-t verra moins de bruit que prévu,
    donc fera peu de changements — puis raffinera normalement sur les bas-t.
    """
    ab = schedule["alpha_bar"].to(device)
    T  = schedule.T

    timesteps = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long).to(device)
    ts_cur  = timesteps[:-1]
    ts_prev = timesteps[1:]

    z = z_start.clone()
    B = z.shape[0]

    for t_cur, t_prev in zip(ts_cur, ts_prev):
        t_batch = torch.full((B,), t_cur.item(), device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(z, t_batch, z0, z1)

        a_cur  = ab[t_cur]
        a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        x0_pred = (z - (1 - a_cur).sqrt() * eps) / a_cur.sqrt().clamp(min=1e-6)
        x0_pred = x0_pred.clamp(-10, 10)
        dir_xt  = (1 - a_prev).clamp(0).sqrt() * eps
        z       = a_prev.sqrt() * x0_pred + dir_xt

    return z


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, required=True)
    p.add_argument("--data_root", type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--out_dir",   type=str, default="eval_warmstart")
    p.add_argument("--t_start",   type=int, default=50,  help="Timestep de départ (bruit ajouté)")
    p.add_argument("--steps",     type=int, default=50,  help="Nombre de steps DDIM")
    p.add_argument("--full_schedule", action="store_true", help="Schedule complet T-1→0 depuis z_noisy")
    p.add_argument("--pred_type", type=str, default="v", choices=["eps", "v"], help="v (nouveau modèle) ou eps (ancien)")
    p.add_argument("--T",         type=int, default=200)
    p.add_argument("--base_ch",   type=int, default=64)
    p.add_argument("--vae_id",    type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--n",         type=int, default=10)
    p.add_argument("--duration",  type=int, default=500)
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | t_start={args.t_start} | steps={args.steps} | T={args.T}")

    print("Chargement VAE...")
    vae = FrozenVAE(model_id=args.vae_id).to(device)

    schedule = CosineNoiseSchedule(T=args.T).to(device)

    print(f"Chargement modèle: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = DiffusionUNet(latent_ch=4, base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Epoch {ckpt['epoch'] + 1}")

    _, val_loader, _ = get_dataloaders(
        args.data_root,
        crop_size=256,
        batch_size=1,
        num_workers=0,
    )

    psnr_list = []
    ssim_list = []

    print(f"\nGeneration de {args.n} GIFs (warm start t={args.t_start})...")
    for i, batch in enumerate(val_loader):
        if i >= args.n:
            break

        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        z0   = vae.encode(I0)
        z1   = vae.encode(I1)
        z_gt = vae.encode(Imid)

        # Bruiter z_gt au timestep t_start
        t_tensor = torch.tensor([args.t_start], device=device, dtype=torch.long)
        z_noisy, noise_added = schedule.q_sample(z_gt, t_tensor)

        # DDIM via DDIMSampler (gère v-prediction et warmstart)
        sampler = DDIMSampler(model, schedule, T_inf=args.steps, eta=0.0,
                              prediction_type=args.pred_type)
        if args.full_schedule:
            z_denoised = ddim_full_from_noisy(model, schedule, z_noisy, args.steps, z0, z1, device)
        else:
            z_denoised = sampler.sample(z0, z1, device=device,
                                        z_start=z_noisy, t_start=args.t_start)

        pred_pixel  = vae.decode(z_denoised)
        noisy_pixel = vae.decode(z_noisy)

        psnr = compute_psnr(pred_pixel, Imid)
        ssim = compute_ssim(pred_pixel.cpu(), Imid.cpu())
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        mode_label = f"full {args.steps}steps" if args.full_schedule else f"t={args.t_start}->0 {args.steps}steps"

        # GIF complet (debug) : I0 | bruité | pred | GT | I1
        frames_full = [
            add_label(tensor_to_pil(I0[0]),          "I0"),
            add_label(tensor_to_pil(noisy_pixel[0]),  f"Imid noisy t={args.t_start}"),
            add_label(tensor_to_pil(pred_pixel[0]),   f"{mode_label} | PSNR={psnr:.2f}"),
            add_label(tensor_to_pil(Imid[0]),         "Imid GT"),
            add_label(tensor_to_pil(I1[0]),           "I1"),
        ]
        out_full = os.path.join(args.out_dir, f"sample_{i:03d}_full.gif")
        make_gif(frames_full, out_full, duration=args.duration)

        # GIF clean (production) : I0 | pred | I1
        frames_clean = [
            add_label(tensor_to_pil(I0[0]),         "I0"),
            add_label(tensor_to_pil(pred_pixel[0]), f"Imid pred | PSNR={psnr:.2f}"),
            add_label(tensor_to_pil(I1[0]),         "I1"),
        ]
        out_clean = os.path.join(args.out_dir, f"sample_{i:03d}_clean.gif")
        make_gif(frames_clean, out_clean, duration=args.duration)

        print(f"  [{i+1}/{args.n}] PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  -> {out_clean} + {out_full}")

    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    print(f"\nMoyenne sur {len(psnr_list)} samples:")
    print(f"  PSNR : {mean_psnr:.3f} dB")
    print(f"  SSIM : {mean_ssim:.4f}")
    print(f"\nGIFs dans : {args.out_dir}/")


if __name__ == "__main__":
    main()
