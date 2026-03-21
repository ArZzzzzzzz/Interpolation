"""
models/diffusion.py — Latent Diffusion Model for Conditional Frame Interpolation

Training objective (DDPM-style noise prediction):
    1. Sample timestep t ~ Uniform{1, T}
    2. Noise target latent:   z_t = sqrt(alpha_bar_t) * z_gt + sqrt(1-alpha_bar_t) * eps
    3. Condition on (z0, z1): model input is concat([z_t, z0, z1])  [12 channels for 4-ch latents]
    4. Predict noise:         eps_theta(z_t, t, z0, z1)
    5. Loss:                  MSE(eps_pred, eps)

Inference (DDIM for fast sampling):
    Starting from z_T ~ N(0,I), iteratively denoise conditioned on (z0, z1).

Architecture:
    - UNet with sinusoidal timestep embedding injected at each resolution via AdaGN
    - Same encoder-decoder structure but with timestep conditioning
    - Residual blocks with GroupNorm + SiLU

Design choices:
    - Sinusoidal timestep embedding (standard in diffusion models)
    - AdaGN (Adaptive Group Norm) for timestep injection — simple and effective
    - Linear noise schedule (can be switched to cosine)
    - DDIM sampler for fast deterministic inference (50 steps instead of 1000)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """
    Linear beta schedule from beta_start to beta_end over T steps.
    Pre-computes alpha_bar (cumulative product of 1-beta).
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas           = torch.linspace(beta_start, beta_end, T)
        alphas          = 1.0 - betas
        alpha_bar       = torch.cumprod(alphas, dim=0)
        alpha_bar_prev  = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register = {
            "betas":          betas,
            "alphas":         alphas,
            "alpha_bar":      alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "sqrt_alpha_bar":          alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar":(1 - alpha_bar).sqrt(),
        }

    def to(self, device):
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def __getitem__(self, key):
        return self.register[key]

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward diffusion: z_t = sqrt(alpha_bar_t) * z0 + sqrt(1-alpha_bar_t) * eps

        Args:
            z0    : [B, C, h, w]  clean latent
            t     : [B]           timestep indices (0-indexed)
            noise : [B, C, h, w]  optional pre-sampled noise

        Returns:
            z_t   : noised latent
            noise : the noise used
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sa  = self._gather(self.register["sqrt_alpha_bar"],          t, z0)
        sb  = self._gather(self.register["sqrt_one_minus_alpha_bar"], t, z0)
        return sa * z0 + sb * noise, noise

    @staticmethod
    def _gather(coef: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Gather coef[t] and reshape to [B, 1, 1, 1] for broadcasting."""
        out = coef[t]
        return out.view(t.shape[0], *([1] * (ref.ndim - 1)))


class CosineNoiseSchedule(LinearNoiseSchedule):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Maintains more signal at intermediate timesteps compared to linear.
    """
    def __init__(self, T: int = 1000, s: float = 0.008):
        # Override betas with cosine schedule
        t = torch.linspace(0, T, T + 1)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(0, 0.999)

        # Manually set attributes (bypass __init__)
        self.T = T
        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register = {
            "betas":                   betas,
            "alphas":                  alphas,
            "alpha_bar":               alpha_bar,
            "alpha_bar_prev":          alpha_bar_prev,
            "sqrt_alpha_bar":          alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar":(1 - alpha_bar).sqrt(),
        }


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timesteps, then projected to embed_dim.
    Standard in DDPM/DDIM implementations.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : [B]  integer timesteps

        Returns:
            emb : [B, dim*4]
        """
        t = t.float()
        args = t[:, None] * self.freqs[None, :]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Adaptive Group Norm for timestep conditioning
# ---------------------------------------------------------------------------

class AdaGN(nn.Module):
    """
    Adaptive Group Norm: scale and shift feature maps using timestep embedding.
    emb → (scale, shift) applied after GroupNorm.
    """
    def __init__(self, num_groups: int, num_channels: int, emb_dim: int):
        super().__init__()
        self.norm    = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj    = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, C, H, W]
            emb : [B, emb_dim]
        """
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Residual block with timestep embedding
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with AdaGN for timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaGN(min(num_groups, out_ch), out_ch, emb_dim)
        self.norm2 = AdaGN(min(num_groups, out_ch), out_ch, emb_dim)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x), emb))
        h = self.norm2(self.conv2(h), emb)
        return self.act(h + self.skip(x))


# ---------------------------------------------------------------------------
# Diffusion UNet (noise predictor)
# ---------------------------------------------------------------------------

class DiffusionUNet(nn.Module):
    """
    Conditional UNet that predicts the noise epsilon added to z_gt.

    Input channels: latent_ch * 3  (z_t, z0, z1 concatenated)
    Output channels: latent_ch     (predicted noise epsilon)

    Timestep is injected via AdaGN at every residual block.

    Args:
        latent_ch : latent channels (4 for SD VAE)
        base_ch   : base channel count
        t_emb_dim : timestep embedding dim (before projection)
        num_groups: GroupNorm groups
    """

    def __init__(
        self,
        latent_ch: int = 4,
        base_ch:   int = 128,
        t_emb_dim: int = 256,
        num_groups: int = 8,
    ):
        super().__init__()
        in_ch  = latent_ch * 3   # z_t + z0 + z1
        emb_ch = t_emb_dim * 4   # after projection in SinusoidalTimestepEmbedding

        self.t_emb = SinusoidalTimestepEmbedding(t_emb_dim)

        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList([
            ResBlock(ch[0], ch[0], emb_ch, num_groups),
            ResBlock(ch[0], ch[0], emb_ch, num_groups),
        ])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(ch[0], ch[1], emb_ch, num_groups))
        self.enc1  = ResBlock(ch[1], ch[1], emb_ch, num_groups)

        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(ch[1], ch[2], emb_ch, num_groups))
        self.enc2  = ResBlock(ch[2], ch[2], emb_ch, num_groups)

        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(ch[2], ch[3], emb_ch, num_groups))

        # Bottleneck
        self.mid1 = ResBlock(ch[3], ch[3], emb_ch, num_groups)
        self.mid2 = ResBlock(ch[3], ch[3], emb_ch, num_groups)

        # Decoder
        self.up3   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3  = ResBlock(ch[3] + ch[2], ch[2], emb_ch, num_groups)

        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2  = ResBlock(ch[2] + ch[1], ch[1], emb_ch, num_groups)

        self.up1   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1  = ResBlock(ch[1] + ch[0], ch[0], emb_ch, num_groups)

        # Output
        self.out_norm = nn.GroupNorm(min(num_groups, ch[0]), ch[0])
        self.out_conv = nn.Conv2d(ch[0], latent_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _apply_seq(self, blocks, x, emb):
        """Apply a list of ResBlocks sequentially."""
        for b in blocks:
            x = b(x, emb)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t:   torch.Tensor,
        z0:  torch.Tensor,
        z1:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t : [B, 4, h, w]  noised target latent
            t   : [B]            integer timesteps
            z0  : [B, 4, h, w]  condition: frame 0 latent
            z1  : [B, 4, h, w]  condition: frame 1 latent

        Returns:
            eps_pred : [B, 4, h, w]  predicted noise
        """
        emb = self.t_emb(t)                                  # [B, emb_ch]

        x = torch.cat([z_t, z0, z1], dim=1)                 # [B, 12, h, w]
        x = self.in_conv(x)                                  # [B, ch[0], h, w]

        # Encoder
        s0 = self._apply_seq(self.enc_blocks, x, emb)        # [B, ch0, h, w]
        s1 = self.enc1(self.down1[1](self.down1[0](s0), emb) if False
                       else self._down_block(self.down1, s0, emb), emb)
        s2 = self.enc2(self._down_block(self.down2, s1, emb), emb)
        x  = self._down_block(self.down3, s2, emb)

        # Bottleneck
        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        # Decoder
        x = self.up3(x)
        x = self._match_and_cat(x, s2)
        x = self.dec3(x, emb)

        x = self.up2(x)
        x = self._match_and_cat(x, s1)
        x = self.dec2(x, emb)

        x = self.up1(x)
        x = self._match_and_cat(x, s0)
        x = self.dec1(x, emb)

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

    def _down_block(self, block, x, emb):
        """MaxPool → ResBlock with emb."""
        return block[1](block[0](x), emb)

    @staticmethod
    def _match_and_cat(x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([x, skip], dim=1)


# ---------------------------------------------------------------------------
# DDIM Sampler (fast inference)
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    Deterministic DDIM sampler for fast inference with T_inf << T_train steps.

    Reference: Song et al., "Denoising Diffusion Implicit Models", 2021.

    Args:
        model    : DiffusionUNet
        schedule : LinearNoiseSchedule or CosineNoiseSchedule
        T_inf    : number of inference steps (default 50)
        eta      : stochasticity (0 = deterministic DDIM, 1 = DDPM)
    """

    def __init__(self, model: DiffusionUNet, schedule, T_inf: int = 50, eta: float = 0.0,
                 prediction_type: str = "eps"):
        self.model           = model
        self.schedule        = schedule
        self.T_inf           = T_inf
        self.eta             = eta
        self.prediction_type = prediction_type  # "eps" ou "v"

        # Select T_inf timesteps, evenly spaced (pour génération depuis bruit pur)
        self.timesteps = torch.linspace(
            schedule.T - 1, 0, T_inf, dtype=torch.long
        )

    @torch.no_grad()
    def sample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        shape: tuple = None,
        device: str = "cuda",
        z_start: torch.Tensor = None,
        t_start: int = None,
    ) -> torch.Tensor:
        """
        Génère z_pred conditionné sur (z0, z1).

        Args:
            z0, z1   : [B, 4, h, w]  condition latents
            shape    : latent shape ; inferred from z0 if None
            device   : target device
            z_start  : [B, 4, h, w]  latent bruité de départ (warmstart)
            t_start  : int            timestep correspondant à z_start

        Si z_start et t_start sont fournis : DDIM de t_start → 0 (warmstart).
        Sinon : DDIM depuis bruit pur (génération complète).
        """
        self.schedule.to(device)
        shape = shape or z0.shape
        ab    = self.schedule["alpha_bar"].to(device)

        if z_start is not None and t_start is not None:
            # Warmstart : T_inf steps de t_start → 0
            z         = z_start
            timesteps = torch.linspace(t_start, 0, self.T_inf + 1, dtype=torch.long).to(device)
            ts_cur    = timesteps[:-1]
            ts_prev   = timesteps[1:]
        else:
            # Génération depuis bruit pur
            z       = torch.randn(shape, device=device)
            ts_cur  = self.timesteps.to(device)
            ts_prev = torch.cat([self.timesteps[1:], torch.zeros(1, dtype=torch.long)]).to(device)

        for t_cur, t_prev in zip(ts_cur, ts_prev):
            t_batch = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.long)
            out     = self.model(z, t_batch, z0, z1)

            a_cur  = ab[t_cur]
            a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            if self.prediction_type == "v":
                # v = sqrt_ab * eps - sqrt_1mab * x0
                # → x0 = sqrt_ab * z - sqrt_1mab * v
                # → eps = sqrt_1mab * z + sqrt_ab * v
                x0_pred = a_cur.sqrt() * z - (1 - a_cur).sqrt() * out
                eps_eq  = (1 - a_cur).sqrt() * z + a_cur.sqrt() * out
            else:
                eps_eq  = out
                x0_pred = (z - (1 - a_cur).sqrt() * eps_eq) / a_cur.sqrt().clamp(min=1e-6)

            x0_pred = x0_pred.clamp(-10, 10)
            dir_xt  = (1 - a_prev - self.eta ** 2 * (1 - a_cur)).clamp(0).sqrt() * eps_eq
            noise   = self.eta * (1 - a_prev).sqrt() * torch.randn_like(z) if self.eta > 0 else 0
            z       = a_prev.sqrt() * x0_pred + dir_xt + noise

        return z


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model    = DiffusionUNet(latent_ch=4, base_ch=128)
    schedule = CosineNoiseSchedule(T=1000)

    n = sum(p.numel() for p in model.parameters())
    print(f"DiffusionUNet parameters: {n:,}")

    B, C, h, w = 2, 4, 32, 32
    z0  = torch.randn(B, C, h, w)
    z1  = torch.randn(B, C, h, w)
    z_gt = torch.randn(B, C, h, w)
    t   = torch.randint(0, 1000, (B,))

    z_t, noise = schedule.q_sample(z_gt, t)
    eps_pred   = model(z_t, t, z0, z1)
    loss       = F.mse_loss(eps_pred, noise)

    print("eps_pred shape:", eps_pred.shape)
    print("Loss:", loss.item())
