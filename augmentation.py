"""
augmentation.py — Temporally coherent augmentation for VFI triplets

Key constraint: augmentation must remain valid across (I0, Imid, I1).
If we independently transform each frame, Imid no longer corresponds to
the true midpoint of the transformed I0/I1 — the supervision becomes wrong.

Solution: sample an affine transform T at t=0 and T' at t=1, then
interpolate at t=0.5 for Imid:
    transform(I0)   = apply(T_0)
    transform(Imid) = apply(lerp(T_0, T_1, 0.5))
    transform(I1)   = apply(T_1)

This simulates a smooth camera or object motion trajectory, making the
triplet harder (larger effective displacement) while remaining physically
coherent — the training signal stays valid.

Supported transforms (all configurable):
    - Translation (tx, ty as fraction of image size)
    - Rotation (degrees)
    - Uniform scaling (zoom)

These are combined into a single affine matrix and applied via
F.affine_grid + F.grid_sample for differentiability and speed.

Usage:
    aug = TemporalAugmentation(degrees=5, translate=0.05, scale_range=(0.95, 1.05), p=0.5)
    I0_aug, Imid_aug, I1_aug = aug(I0, Imid, I1)  # PIL Images

    # Or as tensor transforms (for use inside a dataset __getitem__):
    aug_t = TemporalAugmentationTensor(...)
    I0_t, Imid_t, I1_t = aug_t(I0_tensor, Imid_tensor, I1_tensor)
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def _build_affine_matrix(angle_deg: float, tx: float, ty: float, scale: float) -> torch.Tensor:
    """
    Returns a 2x3 affine matrix for use with F.affine_grid.
    Convention: normalized coords in [-1, 1].
        angle_deg : rotation in degrees (counter-clockwise)
        tx, ty    : translation as fraction of image size
        scale     : uniform zoom (>1 = zoom out, <1 = zoom in)
    """
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Combined: scale * rotation + translation
    m = torch.tensor([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ], dtype=torch.float32)
    return m


def _lerp_params(p0: dict, p1: dict, t: float) -> dict:
    """Linear interpolation of affine parameters at time t in [0, 1]."""
    return {k: p0[k] * (1 - t) + p1[k] * t for k in p0}


def _apply_affine_tensor(img: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Apply affine transform to a float tensor [C, H, W] or [B, C, H, W].
    Returns transformed tensor, same shape.
    """
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    M   = _build_affine_matrix(**params).unsqueeze(0).to(img.device)  # [1, 2, 3]
    M   = M.expand(img.shape[0], -1, -1)
    grd = F.affine_grid(M, img.shape, align_corners=True)
    out = F.grid_sample(img, grd, mode="bilinear", padding_mode="reflection", align_corners=True)

    if squeeze:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# TemporalAugmentation (PIL-based, for use in Dataset __getitem__)
# ---------------------------------------------------------------------------

class TemporalAugmentation:
    """
    Temporally coherent random affine augmentation for VFI triplets.

    Call with PIL Images: aug(I0, Imid, I1) → (I0_aug, Imid_aug, I1_aug)

    Args:
        degrees     : max rotation ± degrees
        translate   : max translation as fraction of image size (tx, ty)
        scale_range : (min_scale, max_scale) for zoom
        p           : probability of applying any augmentation
        flip_p      : probability of horizontal flip (applied consistently)
    """

    def __init__(
        self,
        degrees:     float = 5.0,
        translate:   float = 0.05,
        scale_range: tuple = (0.95, 1.05),
        p:           float = 0.5,
        flip_p:      float = 0.5,
    ):
        self.degrees     = degrees
        self.translate   = translate
        self.scale_range = scale_range
        self.p           = p
        self.flip_p      = flip_p

    def _sample_params(self) -> dict:
        return {
            "angle_deg": random.uniform(-self.degrees, self.degrees),
            "tx":        random.uniform(-self.translate, self.translate),
            "ty":        random.uniform(-self.translate, self.translate),
            "scale":     random.uniform(*self.scale_range),
        }

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img) * 2.0 - 1.0  # [0,1] -> [-1,1]

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        return TF.to_pil_image(((t + 1.0) / 2.0).clamp(0, 1))

    def __call__(
        self,
        I0:   Image.Image,
        Imid: Image.Image,
        I1:   Image.Image,
    ) -> tuple:
        # Optional horizontal flip (same for all three)
        if random.random() < self.flip_p:
            I0, Imid, I1 = TF.hflip(I0), TF.hflip(Imid), TF.hflip(I1)

        # Skip affine with probability (1-p)
        if random.random() > self.p:
            return I0, Imid, I1

        # Sample transform at t=0 and t=1
        p0 = self._sample_params()
        p1 = self._sample_params()
        p_mid = _lerp_params(p0, p1, 0.5)

        # Convert to tensors
        t0   = self._pil_to_tensor(I0)
        tmid = self._pil_to_tensor(Imid)
        t1   = self._pil_to_tensor(I1)

        # Apply affine transforms
        t0   = _apply_affine_tensor(t0,   p0)
        tmid = _apply_affine_tensor(tmid, p_mid)
        t1   = _apply_affine_tensor(t1,   p1)

        return self._tensor_to_pil(t0), self._tensor_to_pil(tmid), self._tensor_to_pil(t1)


# ---------------------------------------------------------------------------
# TemporalAugmentationTensor (tensor-based, faster for GPU training)
# ---------------------------------------------------------------------------

class TemporalAugmentationTensor:
    """
    Same as TemporalAugmentation but operates on float tensors [C, H, W] in [-1, 1].
    Useful when augmentation is applied after ToTensor in the pipeline.
    """

    def __init__(
        self,
        degrees:     float = 5.0,
        translate:   float = 0.05,
        scale_range: tuple = (0.95, 1.05),
        p:           float = 0.5,
        flip_p:      float = 0.5,
    ):
        self.degrees     = degrees
        self.translate   = translate
        self.scale_range = scale_range
        self.p           = p
        self.flip_p      = flip_p

    def _sample_params(self) -> dict:
        return {
            "angle_deg": random.uniform(-self.degrees, self.degrees),
            "tx":        random.uniform(-self.translate, self.translate),
            "ty":        random.uniform(-self.translate, self.translate),
            "scale":     random.uniform(*self.scale_range),
        }

    def __call__(
        self,
        I0:   torch.Tensor,
        Imid: torch.Tensor,
        I1:   torch.Tensor,
    ) -> tuple:
        # Horizontal flip
        if random.random() < self.flip_p:
            I0, Imid, I1 = I0.flip(-1), Imid.flip(-1), I1.flip(-1)

        if random.random() > self.p:
            return I0, Imid, I1

        p0    = self._sample_params()
        p1    = self._sample_params()
        p_mid = _lerp_params(p0, p1, 0.5)

        return (
            _apply_affine_tensor(I0,   p0),
            _apply_affine_tensor(Imid, p_mid),
            _apply_affine_tensor(I1,   p1),
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    aug = TemporalAugmentationTensor(degrees=10, translate=0.1, p=1.0, flip_p=0.5)

    I0   = torch.randn(3, 256, 256)
    Imid = torch.randn(3, 256, 256)
    I1   = torch.randn(3, 256, 256)

    I0_a, Imid_a, I1_a = aug(I0, Imid, I1)
    print("Augmented shapes:", I0_a.shape, Imid_a.shape, I1_a.shape)
    print("Value range:", I0_a.min().item(), I0_a.max().item())
