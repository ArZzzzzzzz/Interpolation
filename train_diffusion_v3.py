"""
train_diffusion_v3.py — Entraînement du modèle de diffusion V3

Architecture : DiffusionUNetV3 + CondEncoder (ControlNet-style)
    - z_t (bruit pur → Imid) conditionné sur z0/z1 injectés à chaque niveau
    - Entraîné from scratch sur Vimeo-90K
    - T=1000, cosine schedule, v-prediction
    - Inférence DDIM depuis bruit pur (pas de warmstart)

Usage:
    python train_diffusion_v3.py \
        --data_root "D:/projet mumltimodal/vimeo_triplet" \
        --epochs 60 --batch_size 4 --save_dir checkpoints/diffusion_v3
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import get_dataloaders
from models.unet_latent import FrozenVAE
from models.diffusion_v3 import DiffusionUNetV3, CosineNoiseSchedule, DDIMSampler
from metrics import compute_psnr, compute_ssim, MetricAccumulator
from losses import CharbonnierLoss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, vae, schedule, loader, optimizer, scaler, device):
    model.train()
    acc  = MetricAccumulator()
    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        with torch.no_grad():
            z0   = vae.encode(I0)
            z1   = vae.encode(I1)
            z_gt = vae.encode(Imid)

        t          = torch.randint(0, schedule.T, (I0.shape[0],), device=device)
        z_t, noise = schedule.q_sample(z_gt, t)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            out = model(z_t, t, z0, z1)

            sqrt_ab   = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1, 1).float()
            sqrt_1mab = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1).float()

            # v-prediction
            v_target = sqrt_ab * noise.float() - sqrt_1mab * z_gt.float()
            loss     = F.mse_loss(out.float(), v_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc.update(loss=loss.item())
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return acc.compute()


# ---------------------------------------------------------------------------
# Validation — DDIM depuis bruit pur
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, vae, schedule, loader, device, T_inf_val=50):
    """Validation rapide avec T_inf_val steps DDIM depuis bruit pur."""
    model.eval()
    sampler = DDIMSampler(model, schedule, T_inf=T_inf_val, eta=0.0, prediction_type="v")
    acc     = MetricAccumulator()

    for i, batch in enumerate(loader):
        if i >= 30:
            break
        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        z0 = vae.encode(I0)
        z1 = vae.encode(I1)

        z_pred     = sampler.sample(z0, z1, device=device)
        pred_pixel = vae.decode(z_pred)

        acc.update(
            psnr=compute_psnr(pred_pixel, Imid),
            ssim=compute_ssim(pred_pixel.cpu(), Imid.cpu()),
        )

    return acc.compute()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   required=True)
    p.add_argument("--vae_id",       type=str,   default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--save_dir",     type=str,   default="checkpoints/diffusion_v3")
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--T",            type=int,   default=1000)
    p.add_argument("--T_inf",        type=int,   default=250,  help="DDIM steps pour evaluation finale")
    p.add_argument("--T_inf_val",    type=int,   default=50,   help="DDIM steps pour validation rapide pendant training")
    p.add_argument("--base_ch",      type=int,   default=128)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--crop_size",    type=int,   default=256)
    p.add_argument("--val_every",    type=int,   default=5)
    p.add_argument("--resume",       type=str,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Loading VAE: {args.vae_id}")
    vae      = FrozenVAE(model_id=args.vae_id).to(device)
    schedule = CosineNoiseSchedule(T=args.T).to(device)

    model = DiffusionUNetV3(latent_ch=4, base_ch=args.base_ch).to(device)
    n_tot  = sum(p.numel() for p in model.parameters())
    n_cond = sum(p.numel() for p in model.cond_enc.parameters())
    print(f"DiffusionUNetV3 : {n_tot:,} params ({n_cond:,} CondEncoder + {n_tot-n_cond:,} Denoiser)")
    print(f"T={args.T} | T_inf_val={args.T_inf_val} | T_inf_final={args.T_inf} | pred=v-prediction")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    start_epoch = 0
    best_psnr   = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr   = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs):
        t0      = time.time()
        train_m = train_one_epoch(model, vae, schedule, train_loader, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | LR: {lr:.2e} | "
            f"loss: {train_m['loss']:.5f} | Time: {elapsed:.1f}s",
            end="",
        )

        if (epoch + 1) % args.val_every == 0:
            val_m = validate(model, vae, schedule, val_loader, device, T_inf_val=args.T_inf_val)
            print(f" | Val PSNR: {val_m['psnr']:.3f} dB | SSIM: {val_m['ssim']:.4f}", end="")

            if val_m["psnr"] > best_psnr:
                best_psnr = val_m["psnr"]
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_psnr": best_psnr, "args": vars(args),
                }, os.path.join(args.save_dir, "best.pth"))
                print(" -> best saved", end="")

        print()

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr, "args": vars(args),
            }, os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth"))

    print(f"\nDone. Best val PSNR: {best_psnr:.3f} dB")


if __name__ == "__main__":
    main()
