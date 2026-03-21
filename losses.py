"""
losses.py — Loss functions for Video Frame Interpolation

Implemented:
  - Charbonnier loss     : smooth L1 variant, robust to outliers
  - Perceptual loss      : VGG16 feature-space distance
  - Combined pixel loss  : Charbonnier + weighted perceptual
  - Latent loss          : L1 in latent space + Charbonnier in pixel space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


# ---------------------------------------------------------------------------
# Charbonnier loss
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((x-y)^2 + eps^2)
    A smooth approximation of L1 that is differentiable at 0.
    More robust to outliers than L2, less sensitive to small errors than L1.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        return loss.mean()


# ---------------------------------------------------------------------------
# Perceptual loss (VGG16)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Feature-space perceptual loss using VGG16 relu layers.

    Extracts features at relu1_2, relu2_2, relu3_3 and computes
    L1 distance between feature maps of prediction and target.

    Inputs expected in [-1, 1]; internally renormalised to ImageNet stats.
    """

    # ImageNet mean/std for VGG normalisation
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, layer_weights: dict = None):
        super().__init__()

        # Default: equal weight on three early-mid VGG layers
        self.layer_weights = layer_weights or {
            "relu1_2": 1.0,
            "relu2_2": 1.0,
            "relu3_3": 1.0,
        }

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        # Slice VGG into named stages
        # relu1_2 = layers 0-3 (conv1_1, relu, conv1_2, relu)
        # relu2_2 = layers 4-8
        # relu3_3 = layers 9-15
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Map from [-1,1] to ImageNet normalisation."""
        x = (x + 1.0) / 2.0              # [0, 1]
        mean = self._MEAN.to(x.device)
        std  = self._STD.to(x.device)
        return (x - mean) / std

    def _extract_features(self, x: torch.Tensor) -> dict:
        x = self._normalise(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return {"relu1_2": f1, "relu2_2": f2, "relu3_3": f3}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats   = self._extract_features(pred)
        target_feats = self._extract_features(target)

        loss = 0.0
        for layer, w in self.layer_weights.items():
            loss = loss + w * F.l1_loss(pred_feats[layer], target_feats[layer])
        return loss


# ---------------------------------------------------------------------------
# Combined pixel loss (used by pixel-space UNet)
# ---------------------------------------------------------------------------

class PixelLoss(nn.Module):
    """
    Combined loss for pixel-space models:
        L = Charbonnier(pred, target) + lambda_perc * Perceptual(pred, target)

    Args:
        lambda_perc : weight for perceptual term (0 disables it)
    """
    def __init__(self, lambda_perc: float = 0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.lambda_perc = lambda_perc
        if lambda_perc > 0:
            self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        charb = self.charbonnier(pred, target)
        losses = {"charbonnier": charb, "total": charb}

        if self.lambda_perc > 0:
            perc = self.perceptual(pred, target)
            losses["perceptual"] = perc
            losses["total"] = charb + self.lambda_perc * perc

        return losses


# ---------------------------------------------------------------------------
# Latent loss (used by latent-space UNet)
# ---------------------------------------------------------------------------

class LatentLoss(nn.Module):
    """
    Combined loss for latent-space models:
        L = lambda_lat * L1(z_pred, z_gt)
          + lambda_pix * Charbonnier(D(z_pred), Imid)
          + lambda_perc * Perceptual(D(z_pred), Imid)    [optional]

    The pixel-space terms require the decoder D to be called externally
    and the decoded image passed as `pred_pixel`.

    Args:
        lambda_lat  : weight on latent L1
        lambda_pix  : weight on pixel Charbonnier
        lambda_perc : weight on perceptual (0 disables)
    """
    def __init__(
        self,
        lambda_lat: float  = 1.0,
        lambda_pix: float  = 1.0,
        lambda_perc: float = 0.1,
    ):
        super().__init__()
        self.lambda_lat  = lambda_lat
        self.lambda_pix  = lambda_pix
        self.lambda_perc = lambda_perc

        self.charbonnier = CharbonnierLoss()
        if lambda_perc > 0:
            self.perceptual = PerceptualLoss()

    def forward(
        self,
        z_pred: torch.Tensor,
        z_gt: torch.Tensor,
        pred_pixel: torch.Tensor = None,
        target_pixel: torch.Tensor = None,
    ) -> dict:
        lat_loss = F.l1_loss(z_pred, z_gt)
        losses = {
            "latent_l1": lat_loss,
            "total": self.lambda_lat * lat_loss,
        }

        if pred_pixel is not None and target_pixel is not None:
            pix_loss = self.charbonnier(pred_pixel, target_pixel)
            losses["pixel_charbonnier"] = pix_loss
            losses["total"] = losses["total"] + self.lambda_pix * pix_loss

            if self.lambda_perc > 0:
                perc_loss = self.perceptual(pred_pixel, target_pixel)
                losses["perceptual"] = perc_loss
                losses["total"] = losses["total"] + self.lambda_perc * perc_loss

        return losses
