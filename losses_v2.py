"""
losses_v2.py — Loss functions for UNetPixelV2

Loss = lambda_charb  * L_charbonnier
     + lambda_perc   * L_perceptual
     + lambda_grad   * L_gradient
     + lambda_motion * L_motion_weighted

- Charbonnier : smooth L1 variant, robust to outliers, default reconstruction loss
- Perceptual  : VGG16 feature-space loss (relu2_2 + relu3_3), penalizes structural blur
- Gradient    : Sobel edge preservation, pushes the model to maintain sharp boundaries
- Motion-weighted : Charbonnier weighted by |I1-I0| — harder motion regions count more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Charbonnier loss
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    L_charb = mean( sqrt( (pred - target)^2 + eps^2 ) )

    Smoother than L1 near zero, more robust than L2 to large errors.
    Standard choice for image reconstruction tasks.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


# ---------------------------------------------------------------------------
# Perceptual loss (VGG16)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Feature-space loss using VGG16 (pretrained, frozen).

    Compares relu2_2 and relu3_3 activations between pred and target.
    Encourages perceptually sharper output — inspired by SRGAN / Johnson et al.

    Inputs expected in [-1, 1]; internally renormalized to ImageNet stats.
    """
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, layers: list = ("relu2_2", "relu3_3")):
        super().__init__()
        vgg  = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
        # relu2_2 = up to index 9, relu3_3 = up to index 16
        cuts = {"relu2_2": 9, "relu3_3": 16}
        cuts = {k: cuts[k] for k in layers}
        self.slices = nn.ModuleDict({
            name: nn.Sequential(*list(vgg.children())[:idx + 1])
            for name, idx in cuts.items()
        })
        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] → ImageNet-normalized."""
        x = (x + 1.0) / 2.0          # [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n   = self._normalize(pred)
        target_n = self._normalize(target)
        loss = 0.0
        for name, slice_ in self.slices.items():
            f_pred   = slice_(pred_n)
            f_target = slice_(target_n).detach()
            loss     = loss + F.l1_loss(f_pred, f_target)
        return loss / len(self.slices)


# ---------------------------------------------------------------------------
# Gradient loss (Sobel)
# ---------------------------------------------------------------------------

class GradientLoss(nn.Module):
    """
    Edge-preserving loss using Sobel filters in x and y directions.

    Penalizes differences in image gradients between pred and target,
    which pushes the model to preserve sharp edges and avoid blur.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.T
        # Shape: [out_ch, in_ch, kH, kW] — applied per-channel via groups
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _sobel(self, x: torch.Tensor) -> tuple:
        """Returns (gx, gy) gradient maps for a [B, C, H, W] tensor."""
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.kx, padding=1).view(B, C, H, W)
        gy = F.conv2d(x_flat, self.ky, padding=1).view(B, C, H, W)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p, gy_p = self._sobel(pred)
        gx_t, gy_t = self._sobel(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


# ---------------------------------------------------------------------------
# Motion-weighted loss
# ---------------------------------------------------------------------------

class MotionWeightedLoss(nn.Module):
    """
    Charbonnier loss weighted by the motion magnitude |I1 - I0|.

    Regions with large motion get higher loss weight, forcing the model
    to pay more attention to hard displacement cases.

    weight = 1 + alpha * normalize(|I1 - I0|)
    """
    def __init__(self, eps: float = 1e-3, alpha: float = 2.0):
        super().__init__()
        self.charb = CharbonnierLoss(eps=eps)
        self.alpha = alpha

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        I0:     torch.Tensor,
        I1:     torch.Tensor,
    ) -> torch.Tensor:
        motion = (I1 - I0).abs().mean(dim=1, keepdim=True)   # [B, 1, H, W]
        motion = motion / (motion.amax(dim=(2, 3), keepdim=True) + 1e-6)  # normalize per image
        weight = (1.0 + self.alpha * motion).expand_as(pred)
        return self.charb(pred, target, weight=weight)


# ---------------------------------------------------------------------------
# Combined loss for UNetPixelV2
# ---------------------------------------------------------------------------

class VFILossV2(nn.Module):
    """
    Combined VFI loss:
        L = lambda_charb  * Charbonnier(pred, gt)
          + lambda_perc   * Perceptual(pred, gt)
          + lambda_grad   * Gradient(pred, gt)
          + lambda_motion * MotionWeighted(pred, gt, I0, I1)

    For multi-scale output (dict with "full"/"half"/"quarter"), losses are
    computed at each scale with decaying weights (1.0, 0.5, 0.25).

    Args:
        lambda_charb  : weight for Charbonnier loss
        lambda_perc   : weight for perceptual loss (0 = disabled, saves VRAM)
        lambda_grad   : weight for gradient loss
        lambda_motion : weight for motion-weighted loss
    """

    def __init__(
        self,
        lambda_charb:  float = 1.0,
        lambda_perc:   float = 0.1,
        lambda_grad:   float = 0.1,
        lambda_motion: float = 0.1,
    ):
        super().__init__()
        self.lambda_charb  = lambda_charb
        self.lambda_perc   = lambda_perc
        self.lambda_grad   = lambda_grad
        self.lambda_motion = lambda_motion

        self.charb  = CharbonnierLoss()
        self.grad   = GradientLoss()
        self.motion = MotionWeightedLoss()
        if lambda_perc > 0:
            self.perc = PerceptualLoss()
        else:
            self.perc = None

    def _loss_at_scale(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        I0:     torch.Tensor,
        I1:     torch.Tensor,
    ) -> torch.Tensor:
        loss = self.lambda_charb * self.charb(pred, target)

        if self.perc is not None and self.lambda_perc > 0:
            loss = loss + self.lambda_perc * self.perc(pred, target)

        if self.lambda_grad > 0:
            loss = loss + self.lambda_grad * self.grad(pred, target)

        if self.lambda_motion > 0:
            loss = loss + self.lambda_motion * self.motion(pred, target, I0, I1)

        return loss

    def forward(
        self,
        pred:   torch.Tensor | dict,
        target: torch.Tensor,
        I0:     torch.Tensor,
        I1:     torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(pred, dict):
            # Multi-scale: full=1.0, half=0.5, quarter=0.25
            scale_weights = {"full": 1.0, "half": 0.5, "quarter": 0.25}
            total = torch.tensor(0.0, device=target.device)
            for key, w in scale_weights.items():
                if key not in pred:
                    continue
                p = pred[key]
                t = F.interpolate(target, size=p.shape[2:], mode="bilinear", align_corners=True)
                i0 = F.interpolate(I0,     size=p.shape[2:], mode="bilinear", align_corners=True)
                i1 = F.interpolate(I1,     size=p.shape[2:], mode="bilinear", align_corners=True)
                total = total + w * self._loss_at_scale(p, t, i0, i1)
            return total
        else:
            return self._loss_at_scale(pred, target, I0, I1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 3, 128, 128
    pred   = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    I0     = torch.randn(B, C, H, W)
    I1     = torch.randn(B, C, H, W)

    loss_fn = VFILossV2(lambda_charb=1.0, lambda_perc=0.1, lambda_grad=0.1, lambda_motion=0.1)
    loss = loss_fn(pred, target, I0, I1)
    print(f"Single-scale loss: {loss.item():.4f}")

    pred_ms = {
        "full":    pred,
        "half":    torch.randn(B, C, H // 2, W // 2),
        "quarter": torch.randn(B, C, H // 4, W // 4),
    }
    loss_ms = loss_fn(pred_ms, target, I0, I1)
    print(f"Multi-scale loss:  {loss_ms.item():.4f}")
