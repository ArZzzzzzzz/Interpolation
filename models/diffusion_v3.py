"""
models/diffusion_v3.py — Latent Diffusion V3 avec CondEncoder (ControlNet-style)

Différence vs diffusion.py (V1/V2) :
    V1/V2 : cat(z_t, z0, z1) → UNet → eps_pred
            z0/z1 injectés UNE SEULE FOIS à l'entrée, signal dilué en profondeur.

    V3    : z_t seul → UNet (débruiteur)
            cat(z0, z1) → CondEncoder → features à 4 échelles
            Ces features sont additionnées aux activations du décodeur à chaque niveau.
            → le conditionnement I0/I1 est présent à CHAQUE résolution.

Architecture :
    CondEncoder  : cat(z0,z1)=8ch → [cond0, cond1, cond2, cond_mid] (4 échelles)
    DiffUNetV3   : z_t (4ch) + timestep AdaGN + injection CondEncoder dans le décodeur

Forward :
    model(z_t, t, z0, z1) → eps_pred   ← même signature que V1/V2 pour compatibilité DDIMSampler

Training : T=1000, v-prediction, cosine schedule
Inference : DDIM 250 steps depuis bruit pur z_T ~ N(0,I)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule (réutilisé depuis diffusion.py)
# ---------------------------------------------------------------------------

class CosineNoiseSchedule:
    def __init__(self, T: int = 1000, s: float = 0.008):
        t = torch.linspace(0, T, T + 1)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)
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

    def to(self, device):
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def __getitem__(self, key):
        return self.register[key]

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sa = self._gather(self.register["sqrt_alpha_bar"],          t, z0)
        sb = self._gather(self.register["sqrt_one_minus_alpha_bar"], t, z0)
        return sa * z0 + sb * noise, noise

    @staticmethod
    def _gather(coef, t, ref):
        return coef[t].view(t.shape[0], *([1] * (ref.ndim - 1)))


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        half  = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        t    = t.float()
        args = t[:, None] * self.freqs[None, :]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Briques de base
# ---------------------------------------------------------------------------

def make_gn(ch, g=8):
    return nn.GroupNorm(min(g, ch), ch)


class AdaGN(nn.Module):
    """GroupNorm modulé par le timestep embedding."""
    def __init__(self, ch, emb_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, ch), ch, affine=False)
        self.proj = nn.Linear(emb_dim, ch * 2)

    def forward(self, x, emb):
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """ResBlock avec timestep via AdaGN — pour le débruiteur."""
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaGN(out_ch, emb_dim, num_groups)
        self.norm2 = AdaGN(out_ch, emb_dim, num_groups)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x), emb))
        h = self.norm2(self.conv2(h), emb)
        return self.act(h + self.skip(x))


class SimpleResBlock(nn.Module):
    """ResBlock sans timestep — pour le CondEncoder."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = make_gn(out_ch, num_groups)
        self.norm2 = make_gn(out_ch, num_groups)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


# ---------------------------------------------------------------------------
# CondEncoder — encode z0/z1 à 4 échelles
# ---------------------------------------------------------------------------

class CondEncoder(nn.Module):
    """
    Encodeur de conditionnement pour cat(z0, z1) = 8ch.

    Produit des features à 4 échelles qui seront injectées dans
    chaque niveau du décodeur du débruiteur :
        cond0   : [B, ch0, H,   W  ]   → injecté dans dec1
        cond1   : [B, ch1, H/2, W/2]   → injecté dans dec2
        cond2   : [B, ch2, H/4, W/4]   → injecté dans dec3
        cond_mid: [B, ch3, H/8, W/8]   → injecté dans le bottleneck

    Zéro-init des projections de sortie → n'ajoute rien au début.
    """

    def __init__(self, in_ch: int = 8, base_ch: int = 128, num_groups: int = 8):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.l0  = SimpleResBlock(in_ch, ch[0], num_groups)
        self.d1  = nn.Sequential(nn.MaxPool2d(2), SimpleResBlock(ch[0], ch[1], num_groups))
        self.d2  = nn.Sequential(nn.MaxPool2d(2), SimpleResBlock(ch[1], ch[2], num_groups))
        self.d3  = nn.Sequential(nn.MaxPool2d(2), SimpleResBlock(ch[2], ch[3], num_groups))

        # Projections zero-init pour stabilité au début
        self.out0   = self._zero_conv(ch[0])
        self.out1   = self._zero_conv(ch[1])
        self.out2   = self._zero_conv(ch[2])
        self.out_mid = self._zero_conv(ch[3])

    @staticmethod
    def _zero_conv(ch):
        c = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(c.weight)
        nn.init.zeros_(c.bias)
        return c

    def forward(self, z0: torch.Tensor, z1: torch.Tensor):
        x  = torch.cat([z0, z1], dim=1)   # [B, 8, H, W]
        f0 = self.l0(x)                   # [B, ch0, H,   W  ]
        f1 = self.d1(f0)                  # [B, ch1, H/2, W/2]
        f2 = self.d2(f1)                  # [B, ch2, H/4, W/4]
        f3 = self.d3(f2)                  # [B, ch3, H/8, W/8]
        return self.out0(f0), self.out1(f1), self.out2(f2), self.out_mid(f3)


# ---------------------------------------------------------------------------
# DiffusionUNetV3 — débruiteur avec injection CondEncoder
# ---------------------------------------------------------------------------

class DiffusionUNetV3(nn.Module):
    """
    UNet de débruitage conditionné sur I0/I1 via CondEncoder.

    Forward : model(z_t, t, z0, z1) → eps_pred
    (même signature que V1/V2 pour compatibilité DDIMSampler)

    En interne :
        - CondEncoder(z0, z1) → 4 feature maps
        - UNet(z_t, t) avec injection additive à chaque niveau du décodeur
    """

    def __init__(
        self,
        latent_ch:  int = 4,
        base_ch:    int = 128,
        t_emb_dim:  int = 256,
        num_groups: int = 8,
    ):
        super().__init__()
        emb_ch = t_emb_dim * 4
        ch     = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Timestep embedding
        self.t_emb = SinusoidalTimestepEmbedding(t_emb_dim)

        # Conditioning encoder
        self.cond_enc = CondEncoder(in_ch=8, base_ch=base_ch, num_groups=num_groups)

        # Encodeur du débruiteur — z_t seulement (4ch)
        self.in_conv = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        self.enc0a   = ResBlock(ch[0], ch[0], emb_ch, num_groups)
        self.enc0b   = ResBlock(ch[0], ch[0], emb_ch, num_groups)

        self.down1  = nn.MaxPool2d(2)
        self.enc1   = ResBlock(ch[0], ch[1], emb_ch, num_groups)
        self.enc1b  = ResBlock(ch[1], ch[1], emb_ch, num_groups)

        self.down2  = nn.MaxPool2d(2)
        self.enc2   = ResBlock(ch[1], ch[2], emb_ch, num_groups)
        self.enc2b  = ResBlock(ch[2], ch[2], emb_ch, num_groups)

        self.down3  = nn.MaxPool2d(2)
        self.enc3   = ResBlock(ch[2], ch[3], emb_ch, num_groups)

        # Bottleneck
        self.mid1 = ResBlock(ch[3], ch[3], emb_ch, num_groups)
        self.mid2 = ResBlock(ch[3], ch[3], emb_ch, num_groups)

        # Décodeur avec injection CondEncoder
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ResBlock(ch[3] + ch[2], ch[2], emb_ch, num_groups)

        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ResBlock(ch[2] + ch[1], ch[1], emb_ch, num_groups)

        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ResBlock(ch[1] + ch[0], ch[0], emb_ch, num_groups)

        # Sortie
        self.out_norm = nn.GroupNorm(min(num_groups, ch[0]), ch[0])
        self.out_conv = nn.Conv2d(ch[0], latent_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_t, t, z0, z1):
        emb = self.t_emb(t)                            # [B, emb_ch]

        # Conditioning features
        cond0, cond1, cond2, cond_mid = self.cond_enc(z0, z1)

        # Encodeur
        x  = self.in_conv(z_t)
        s0 = self.enc0b(self.enc0a(x, emb), emb)      # [B, ch0, H,   W  ]
        s1 = self.enc1b(self.enc1(self.down1(s0), emb), emb)   # [B, ch1, H/2, W/2]
        s2 = self.enc2b(self.enc2(self.down2(s1), emb), emb)   # [B, ch2, H/4, W/4]
        x  = self.enc3(self.down3(s2), emb)            # [B, ch3, H/8, W/8]

        # Bottleneck + injection cond_mid
        x = self.mid1(x, emb)
        x = x + self._align(cond_mid, x)
        x = self.mid2(x, emb)

        # Décodeur + injection CondEncoder (après ResBlock, canaux alignés)
        x = self._cat_skip(self.up3(x), s2)
        x = self.dec3(x, emb)
        x = x + self._align(cond2, x)   # ch[2] == ch[2]

        x = self._cat_skip(self.up2(x), s1)
        x = self.dec2(x, emb)
        x = x + self._align(cond1, x)   # ch[1] == ch[1]

        x = self._cat_skip(self.up1(x), s0)
        x = self.dec1(x, emb)
        x = x + self._align(cond0, x)   # ch[0] == ch[0]

        return self.out_conv(F.silu(self.out_norm(x)))

    @staticmethod
    def _align(src, ref):
        """Resize src to match ref spatial dims if needed."""
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=True)
        return src

    @staticmethod
    def _cat_skip(x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([x, skip], dim=1)


# ---------------------------------------------------------------------------
# DDIM Sampler (compatible V3 — même signature forward)
# ---------------------------------------------------------------------------

class DDIMSampler:
    def __init__(self, model, schedule, T_inf=250, eta=0.0, prediction_type="v"):
        self.model           = model
        self.schedule        = schedule
        self.T_inf           = T_inf
        self.eta             = eta
        self.prediction_type = prediction_type
        self.timesteps       = torch.linspace(schedule.T - 1, 0, T_inf, dtype=torch.long)

    @torch.no_grad()
    def sample(self, z0, z1, shape=None, device="cuda"):
        """Génère z_pred depuis bruit pur, conditionné sur (z0, z1)."""
        self.schedule.to(device)
        shape = shape or z0.shape
        ab    = self.schedule["alpha_bar"].to(device)

        z       = torch.randn(shape, device=device)
        ts_cur  = self.timesteps.to(device)
        ts_prev = torch.cat([self.timesteps[1:], torch.zeros(1, dtype=torch.long)]).to(device)

        for t_cur, t_prev in zip(ts_cur, ts_prev):
            t_batch = torch.full((shape[0],), t_cur.item(), device=device, dtype=torch.long)
            out     = self.model(z, t_batch, z0, z1)

            a_cur  = ab[t_cur]
            a_prev = ab[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            if self.prediction_type == "v":
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
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model    = DiffusionUNetV3(latent_ch=4, base_ch=128)
    schedule = CosineNoiseSchedule(T=1000)

    n_main = sum(p.numel() for p in model.parameters() if p not in set(model.cond_enc.parameters()))
    n_cond = sum(p.numel() for p in model.cond_enc.parameters())
    n_tot  = sum(p.numel() for p in model.parameters())
    print(f"DiffusionUNetV3 : {n_tot:,} params total")
    print(f"  Denoiser      : {n_tot - n_cond:,}")
    print(f"  CondEncoder   : {n_cond:,}")

    B, C, h, w = 2, 4, 32, 32
    z0   = torch.randn(B, C, h, w)
    z1   = torch.randn(B, C, h, w)
    z_gt = torch.randn(B, C, h, w)
    t    = torch.randint(0, 1000, (B,))

    z_t, noise = schedule.q_sample(z_gt, t)
    eps_pred   = model(z_t, t, z0, z1)
    loss = F.mse_loss(eps_pred, noise)
    print(f"eps_pred: {eps_pred.shape} | loss: {loss.item():.4f}")
