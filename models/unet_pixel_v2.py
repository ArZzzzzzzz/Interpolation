"""
models/unet_pixel_v2.py — UNetPixelV2: improved pixel-space VFI model

Architectural choices and motivation:
--------------------------------------
1. 12-ch input [I0, I1, diff, absdiff]
   Giving the model explicit motion cues (diff = I1-I0, absdiff = |I1-I0|) lets it
   focus on *where* motion happens rather than having to discover it implicitly.
   This directly reduces the "conditional average" effect in still regions.

2. GroupNorm + SiLU everywhere (no BatchNorm)
   GroupNorm is statistics-independent across the batch, which matters for
   small batches and reconstruction tasks (each image is its own sample).
   SiLU (Swish) provides smoother gradients than ReLU, common in modern UNets.

3. Residual prediction: output = 0.5*(I0+I1) + UNet(x)
   The average frame is a good low-frequency baseline. The UNet only learns
   the correction, which is smaller and easier to predict stably.
   Configurable via use_residual flag.

4. No final tanh — raw conv output
   Tanh squashes gradients near saturation. We skip it and clamp at eval time.

5. Multi-scale auxiliary heads (optional)
   Decoder outputs at quarter and half resolution support coarse-to-fine losses,
   helping the network learn large motion at low resolution first.

6. Lightweight RefinementNet
   A 4-layer CNN takes [I0, I1, coarse] and predicts a residual correction,
   recovering fine details that the main UNet smooths over.

7. CoordConv channels (optional)
   Absolute spatial position as extra input channels, useful if the model
   needs to disambiguate scene-specific motion patterns.

These choices align with U-Net for dense prediction (skip connections for
local detail), perceptual/GAN-style training philosophy (no averaging losses
alone), and GroupNorm as used in modern generative backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with safe group count (never more groups than channels)."""
    return nn.GroupNorm(min(num_groups, num_channels), num_channels)


class ConvGNSiLU(nn.Module):
    """Conv2d -> GroupNorm -> SiLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            make_gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block: two ConvGNSiLU with skip connection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvGNSiLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
        )
        self.act  = nn.SiLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class DownBlock(nn.Module):
    """MaxPool2d -> ResBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.res  = ResBlock(in_ch, out_ch)

    def forward(self, x):
        return self.res(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample -> cat(skip) -> ResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res = ResBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.res(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Refinement network
# ---------------------------------------------------------------------------

class RefinementNet(nn.Module):
    """
    Lightweight 4-layer CNN that refines a coarse prediction.

    Input : [I0, I1, coarse]  →  9 channels
    Output: residual correction (3 channels), initialized to zero

    The refinement focuses on high-frequency detail recovery that the main
    UNet tends to smooth away due to its multi-scale bottleneck.
    """
    def __init__(self, in_ch: int = 9, mid_ch: int = 64, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNSiLU(in_ch, mid_ch),
            ConvGNSiLU(mid_ch, mid_ch),
            ConvGNSiLU(mid_ch, mid_ch // 2),
            nn.Conv2d(mid_ch // 2, out_ch, 3, padding=1),
        )
        # Zero-init: at the start refine contributes nothing
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([I0, I1, coarse], dim=1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class UNetPixelV2(nn.Module):
    """
    Improved pixel-space frame interpolation UNet.

    Args:
        base_ch        : base channel count (default 64)
        use_residual   : predict residual over 0.5*(I0+I1) (default True)
        use_multiscale : return auxiliary quarter/half-res outputs during training
        use_coordconv  : append normalized x/y coord channels to input
        use_refine     : apply lightweight RefinementNet after main UNet

    Forward:
        I0, I1 : [B, 3, H, W]  input frames in [-1, 1]

    Returns (training, use_multiscale=True):
        dict {"full": [B,3,H,W], "half": [B,3,H/2,W/2], "quarter": [B,3,H/4,W/4]}

    Returns (default / eval):
        [B, 3, H, W]  — clamp to [-1,1] at eval time if needed
    """

    def __init__(
        self,
        base_ch:        int  = 64,
        use_residual:   bool = True,
        use_multiscale: bool = False,
        use_coordconv:  bool = False,
        use_refine:     bool = True,
    ):
        super().__init__()
        self.use_residual   = use_residual
        self.use_multiscale = use_multiscale
        self.use_coordconv  = use_coordconv
        self.use_refine     = use_refine

        in_ch = 12 + (2 if use_coordconv else 0)  # I0+I1+diff+absdiff [+ coords]
        ch    = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Encoder
        self.in_conv = ResBlock(in_ch, ch[0])
        self.down1   = DownBlock(ch[0], ch[1])
        self.down2   = DownBlock(ch[1], ch[2])
        self.down3   = DownBlock(ch[2], ch[3])

        # Bottleneck
        self.mid = nn.Sequential(ResBlock(ch[3], ch[3]), ResBlock(ch[3], ch[3]))

        # Decoder
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])

        # Main output head — zero-init for stable residual training
        self.out_conv = nn.Conv2d(ch[0], 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # Auxiliary multi-scale heads (quarter and half resolution)
        if use_multiscale:
            self.aux_quarter = nn.Conv2d(ch[2], 3, 1)
            self.aux_half    = nn.Conv2d(ch[1], 3, 1)

        # Refinement
        if use_refine:
            self.refine = RefinementNet(in_ch=9, mid_ch=64)

    def _coord_channels(self, B: int, H: int, W: int, device) -> torch.Tensor:
        """Normalized [-1,1] x/y coordinate maps: [B, 2, H, W]."""
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([xs, ys], dim=1)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor):
        B, _, H, W = I0.shape
        diff    = I1 - I0
        absdiff = diff.abs()

        x = torch.cat([I0, I1, diff, absdiff], dim=1)       # [B, 12, H, W]
        if self.use_coordconv:
            x = torch.cat([x, self._coord_channels(B, H, W, I0.device)], dim=1)

        # Encoder
        s0 = self.in_conv(x)   # [B, ch0, H,   W  ]
        s1 = self.down1(s0)    # [B, ch1, H/2, W/2]
        s2 = self.down2(s1)    # [B, ch2, H/4, W/4]
        s3 = self.down3(s2)    # [B, ch3, H/8, W/8]

        # Bottleneck
        m = self.mid(s3)

        # Decoder
        d3 = self.up3(m,  s2)  # [B, ch2, H/4, W/4]
        d2 = self.up2(d3, s1)  # [B, ch1, H/2, W/2]
        d1 = self.up1(d2, s0)  # [B, ch0, H,   W  ]

        # Main prediction
        raw = self.out_conv(d1)
        if self.use_residual:
            base   = 0.5 * (I0 + I1)
            coarse = base + raw
        else:
            coarse = raw

        # Refinement
        if self.use_refine:
            final = coarse + self.refine(I0, I1, coarse)
        else:
            final = coarse

        # Multi-scale auxiliary outputs (training only)
        if self.use_multiscale and self.training:
            aux_q = self.aux_quarter(d3)
            aux_h = self.aux_half(d2)
            if self.use_residual:
                base_q = F.interpolate(base, size=aux_q.shape[2:], mode="bilinear", align_corners=True)
                base_h = F.interpolate(base, size=aux_h.shape[2:], mode="bilinear", align_corners=True)
                aux_q  = base_q + aux_q
                aux_h  = base_h + aux_h
            return {"full": final, "half": aux_h, "quarter": aux_q}

        return final


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNetPixelV2(base_ch=64, use_residual=True, use_multiscale=True, use_refine=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"UNetPixelV2 parameters: {n:,}")

    I0 = torch.randn(2, 3, 256, 256)
    I1 = torch.randn(2, 3, 256, 256)

    model.train()
    out = model(I0, I1)
    print("train multiscale:", {k: v.shape for k, v in out.items()})

    model.eval()
    out = model(I0, I1)
    print("eval:", out.shape)
