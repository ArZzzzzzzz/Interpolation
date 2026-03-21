"""
models/unet_latent.py — FrozenVAE wrapper for Stable Diffusion VAE

Used by train_diffusion_v3.py, eval_diffusion_v3.py, eval_warmstart.py
to encode/decode images to/from latent space.
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class FrozenVAE(nn.Module):
    """Frozen Stable Diffusion VAE (encode/decode only, no training)."""

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.eval()
        self.scale_factor = 0.18215
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [-1,1] → [B, 4, H/8, W/8] latent."""
        dist = self.vae.encode(x).latent_dist
        return dist.mean * self.scale_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, 4, H/8, W/8] latent → [B, 3, H, W] in [-1,1]."""
        return self.vae.decode(z / self.scale_factor).sample
