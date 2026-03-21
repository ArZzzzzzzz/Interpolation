"""
train_pixel_v2.py — Training loop for UNetPixelV2

Usage:
    python train_pixel_v2.py --data_root "C:/Users/Tom/Documents/projet_multimodal/vimeo_triplet"

Key differences from train_pixel.py:
    - Uses UNetPixelV2 (12ch input, GroupNorm, residual pred, refinement)
    - Uses VFILossV2 (Charbonnier + Perceptual + Gradient + Motion-weighted)
    - Temporal augmentation (coherent affine transforms across I0/Imid/I1)
    - Multi-scale loss support (--multiscale flag)
    - Periodic checkpoint every 5 epochs (not 10)
    - All args saved in checkpoint for reproducibility
"""

import argparse
import itertools
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image

from dataset import get_dataloaders
from models.unet_pixel_v2 import UNetPixelV2
from losses_v2 import VFILossV2
from metrics import compute_psnr, compute_ssim, MetricAccumulator
from augmentation import TemporalAugmentationTensor


# ---------------------------------------------------------------------------
# Debug: save augmented samples as animated GIFs
# ---------------------------------------------------------------------------

def save_aug_debug_gifs(batch: dict, aug: TemporalAugmentationTensor, save_dir: str, epoch: int, n: int = 4):
    """
    Save n animated GIFs (I0 -> Imid -> I1) before and after augmentation.
    Helps visually verify that temporal transforms are coherent.
    """
    gif_dir = os.path.join(save_dir, "aug_debug")
    os.makedirs(gif_dir, exist_ok=True)

    def to_pil(t: torch.Tensor) -> Image.Image:
        t = t.cpu().float().clamp(-1, 1)
        return Image.fromarray(((t + 1.0) / 2.0 * 255).permute(1, 2, 0).byte().numpy())

    for i in range(min(n, batch["I0"].shape[0])):
        I0, Imid, I1 = batch["I0"][i], batch["Imid"][i], batch["I1"][i]

        # Original GIF
        frames_orig = [to_pil(I0), to_pil(Imid), to_pil(I1)]
        frames_orig[0].save(
            os.path.join(gif_dir, f"ep{epoch:03d}_sample{i:02d}_original.gif"),
            save_all=True, append_images=frames_orig[1:], duration=300, loop=0,
        )

        # Augmented GIF (force aug by setting p=1 temporarily)
        a0, am, a1 = aug(I0, Imid, I1)
        frames_aug = [to_pil(a0), to_pil(am), to_pil(a1)]
        frames_aug[0].save(
            os.path.join(gif_dir, f"ep{epoch:03d}_sample{i:02d}_augmented.gif"),
            save_all=True, append_images=frames_aug[1:], duration=300, loop=0,
        )


# ---------------------------------------------------------------------------
# Augmentation wrapper for DataLoader batch
# ---------------------------------------------------------------------------

def augment_batch(batch: dict, aug: TemporalAugmentationTensor, device) -> dict:
    """Apply temporal augmentation to a batch, frame by frame."""
    I0s, Imids, I1s = [], [], []
    for i in range(batch["I0"].shape[0]):
        i0, im, i1 = aug(
            batch["I0"][i].to(device),
            batch["Imid"][i].to(device),
            batch["I1"][i].to(device),
        )
        I0s.append(i0); Imids.append(im); I1s.append(i1)
    batch["I0"]   = torch.stack(I0s)
    batch["Imid"] = torch.stack(Imids)
    batch["I1"]   = torch.stack(I1s)
    return batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_step(model, batch, optimizer, scaler, criterion, device, aug):
    """Single gradient step on one batch dict (I0/Imid/I1)."""
    I0   = batch["I0"].to(device)
    Imid = batch["Imid"].to(device)
    I1   = batch["I1"].to(device)

    if aug is not None:
        I0s, Imids, I1s = [], [], []
        for i in range(I0.shape[0]):
            a0, am, a1 = aug(I0[i], Imid[i], I1[i])
            I0s.append(a0); Imids.append(am); I1s.append(a1)
        I0   = torch.stack(I0s)
        Imid = torch.stack(Imids)
        I1   = torch.stack(I1s)

    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda"):
        pred = model(I0, I1)
        loss = criterion(pred, Imid, I0, I1)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, total_epochs, aug=None, video_iter=None):
    model.train()
    acc  = MetricAccumulator()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False, dynamic_ncols=True)

    for batch in pbar:
        # Vimeo batch
        loss_val = _train_step(model, batch, optimizer, scaler, criterion, device, aug)
        acc.update(total=loss_val)

        # Video batch (interleaved 1:1, no temporal aug — dataset already filters by motion)
        if video_iter is not None:
            vid_batch = next(video_iter)
            loss_vid  = _train_step(model, vid_batch, optimizer, scaler, criterion, device, aug=None)
            acc.update(total=loss_vid)

        pbar.set_postfix(loss=f"{acc.compute()['total']:.4f}")

    return acc.compute()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_acc   = MetricAccumulator()
    metric_acc = MetricAccumulator()

    pbar = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch in pbar:
        I0   = batch["I0"].to(device)
        Imid = batch["Imid"].to(device)
        I1   = batch["I1"].to(device)

        with autocast("cuda"):
            pred = model(I0, I1)
            loss = criterion(pred, Imid, I0, I1)

        psnr = compute_psnr(pred, Imid)
        ssim = compute_ssim(pred.cpu(), Imid.cpu())

        loss_acc.update(total=loss.item())
        metric_acc.update(psnr=psnr, ssim=ssim)
        pbar.set_postfix(psnr=f"{psnr:.2f}")

    return {**loss_acc.compute(), **metric_acc.compute()}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UNetPixelV2 training")
    p.add_argument("--data_root",      type=str,   required=True)
    p.add_argument("--save_dir",       type=str,   default="checkpoints/pixel_v2")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--crop_size",      type=int,   default=256)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--base_ch",        type=int,   default=64)

    # Model flags
    p.add_argument("--no_residual",    action="store_true", help="Disable residual prediction")
    p.add_argument("--multiscale",     action="store_true", help="Enable multi-scale auxiliary losses")
    p.add_argument("--no_refine",      action="store_true", help="Disable refinement head")
    p.add_argument("--coordconv",      action="store_true", help="Add coordinate channels to input")

    # Loss weights
    p.add_argument("--lambda_charb",   type=float, default=1.0)
    p.add_argument("--lambda_perc",    type=float, default=0.1,  help="0 = disabled")
    p.add_argument("--lambda_grad",    type=float, default=0.1)
    p.add_argument("--lambda_motion",  type=float, default=0.1)

    # Augmentation
    p.add_argument("--aug",            action="store_true", help="Enable temporal augmentation")
    p.add_argument("--aug_degrees",    type=float, default=5.0)
    p.add_argument("--aug_translate",  type=float, default=0.05)
    p.add_argument("--aug_p",          type=float, default=0.5)

    # Resume
    p.add_argument("--resume",         type=str,   default=None)

    # Mixed training with real-video dataset
    p.add_argument("--video_root",     type=str,   default=None,  help="Path to real-video folder (interleaved 1:1 with Vimeo)")
    p.add_argument("--video_mad_min",  type=float, default=10.0,  help="MAD threshold for video triplets")
    p.add_argument("--video_stride",   type=int,   default=6)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data — Vimeo
    train_loader, val_loader, _ = get_dataloaders(
        args.data_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Data — triplets pré-extraits (optional, interleaved avec Vimeo)
    video_iter = None
    if args.video_root:
        from gan.extracted_dataset import ExtractedTripletDataset
        video_ds = ExtractedTripletDataset(args.video_root, augment=True)
        video_loader = DataLoader(
            video_ds,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = False,
            drop_last   = True,
        )
        video_iter = itertools.cycle(video_loader)
        print(f"Mixed training: {len(video_ds):,} pre-extracted triplets from {args.video_root}")

    # Model
    model = UNetPixelV2(
        base_ch        = args.base_ch,
        use_residual   = not args.no_residual,
        use_multiscale = args.multiscale,
        use_coordconv  = args.coordconv,
        use_refine     = not args.no_refine,
    ).to(device)

    n = sum(p.numel() for p in model.parameters())
    print(f"UNetPixelV2 | {n:,} parameters")
    print(f"  residual={not args.no_residual} | multiscale={args.multiscale} | "
          f"refine={not args.no_refine} | coordconv={args.coordconv}")

    # Loss
    criterion = VFILossV2(
        lambda_charb  = args.lambda_charb,
        lambda_perc   = args.lambda_perc,
        lambda_grad   = args.lambda_grad,
        lambda_motion = args.lambda_motion,
    ).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # Temporal augmentation
    aug = None
    if args.aug:
        aug = TemporalAugmentationTensor(
            degrees     = args.aug_degrees,
            translate   = args.aug_translate,
            p           = args.aug_p,
            flip_p      = 0.0,  # flip already handled in dataset.py
        )
        print(f"Temporal augmentation: degrees={args.aug_degrees}, translate={args.aug_translate}, p={args.aug_p}")

    start_epoch = 0
    best_psnr   = 0.0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr   = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} | best PSNR so far: {best_psnr:.3f} dB")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Save augmentation debug GIFs at epoch 1 and every 10 epochs
        if aug is not None and (epoch == 0 or (epoch + 1) % 10 == 0):
            sample_batch = next(iter(train_loader))
            save_aug_debug_gifs(sample_batch, aug, args.save_dir, epoch + 1)
            print(f"  [aug debug] GIFs saved to {args.save_dir}/aug_debug/")

        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch + 1, args.epochs, aug, video_iter
        )
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"LR: {lr:.2e} | "
            f"Train loss: {train_m['total']:.4f} | "
            f"Val loss: {val_m['total']:.4f} | "
            f"Val PSNR: {val_m['psnr']:.3f} dB | "
            f"Val SSIM: {val_m['ssim']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_m["psnr"] > best_psnr:
            best_psnr = val_m["psnr"]
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_psnr":   best_psnr,
                "val_metrics": val_m,
                "args":        vars(args),
            }, os.path.join(args.save_dir, "best.pth"))
            print(f"  -> Best PSNR {best_psnr:.3f} dB — checkpoint saved")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "args":      vars(args),
            }, os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth"))

    print(f"\nDone. Best val PSNR: {best_psnr:.3f} dB")


if __name__ == "__main__":
    main()
