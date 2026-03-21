"""
metrics.py — Image quality metrics for VFI evaluation

Implemented:
  - PSNR  : Peak Signal-to-Noise Ratio (dB)
  - SSIM  : Structural Similarity Index (skimage, per-image)
  - LPIPS : Learned Perceptual Image Patch Similarity (VGG, optional)

All functions accept tensors in [-1, 1] with shape [B, 3, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        pred    : predicted image tensor, shape [B, C, H, W] in [-1, 1]
        target  : ground-truth tensor, same shape
        max_val : signal range (1.0 for [-1,1] after rescaling to [0,1])

    Returns:
        Mean PSNR over batch (float, dB)
    """
    # Rescale [-1,1] → [0,1]
    pred   = (pred.clamp(-1, 1) + 1.0) / 2.0
    target = (target.clamp(-1, 1) + 1.0) / 2.0

    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=1)   # [B]
    psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))
    return psnr.mean().item()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index (using skimage, per-image, then averaged).

    Args:
        pred   : [B, 3, H, W] in [-1, 1]
        target : [B, 3, H, W] in [-1, 1]

    Returns:
        Mean SSIM over batch (float, in [0, 1])
    """
    # To numpy, [0, 1], channel-last
    pred_np   = ((pred.detach().cpu().clamp(-1, 1) + 1.0) / 2.0).numpy()
    target_np = ((target.detach().cpu().clamp(-1, 1) + 1.0) / 2.0).numpy()

    batch_ssim = []
    for i in range(pred_np.shape[0]):
        p = pred_np[i].transpose(1, 2, 0)    # [H, W, 3]
        t = target_np[i].transpose(1, 2, 0)
        s = sk_ssim(p, t, data_range=1.0, channel_axis=2)
        batch_ssim.append(s)

    return float(np.mean(batch_ssim))


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

class LPIPSMetric(nn.Module):
    """
    Learned Perceptual Image Patch Similarity using VGG16 features.

    Lighter alternative to the official lpips package — computes
    normalised L2 distance in VGG feature space at multiple layers.

    For the official LPIPS metric install: pip install lpips
    and use LPIPSOfficialMetric (see below).

    Args:
        net : backbone ('vgg' only, for now)
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG16_Weights

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]), # relu3_3
            nn.Sequential(*list(vgg.children())[16:23]),# relu4_3
        ])

        # Learned linear weights (approximate LPIPS calibration)
        # Initialise to uniform; for a proper LPIPS use the official package.
        self.weights = [0.0625, 0.125, 0.25, 0.5]

        # VGG normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _normalise(self, x):
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred   = self._normalise(pred)
        target = self._normalise(target)

        loss = 0.0
        for w, sl in zip(self.weights, self.slices):
            pred   = sl(pred)
            target = sl(target)
            # Normalise feature maps across channel dim, then L2
            p_norm = F.normalize(pred,   dim=1)
            t_norm = F.normalize(target, dim=1)
            loss   = loss + w * (p_norm - t_norm).pow(2).mean()

        return loss.item()


def try_official_lpips():
    """
    Returns the official LPIPS metric if `lpips` is installed, else None.
    Usage:
        metric = try_official_lpips()
        if metric:
            score = metric(pred, target, normalize=True)
    """
    try:
        import lpips
        fn = lpips.LPIPS(net="vgg")
        fn.eval()
        return fn
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """
    Running accumulator for batch-level metrics.

    Usage:
        acc = MetricAccumulator()
        for batch in loader:
            acc.update(psnr=compute_psnr(...), ssim=compute_ssim(...))
        results = acc.compute()
    """
    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._sums[k]   = self._sums.get(k, 0.0)   + float(v)
            self._counts[k] = self._counts.get(k, 0)    + 1

    def compute(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()
