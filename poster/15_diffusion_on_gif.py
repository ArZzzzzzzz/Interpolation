"""
poster/15_diffusion_on_gif.py — Extrait I0/I1 d'un GIF et applique DiffusionUNetV3

Usage:
    cd poster
    python 15_diffusion_on_gif.py
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_latent import FrozenVAE
from models.diffusion_v3 import DiffusionUNetV3, CosineNoiseSchedule, DDIMSampler

GIF_PATH = os.path.join(ROOT, "checkpoints/gan_scratch/samples/ep005_s03.gif")
CKPT     = os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results/diffusion_on_gan")
os.makedirs(OUT_DIR, exist_ok=True)

T      = 1000
T_INF  = 250
BASE_CH = 128


def pil_to_tensor(img):
    """PIL RGB → [1,3,256,256] in [-1,1]."""
    img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t):
    """[1,3,H,W] in [-1,1] → PIL RGB."""
    t = t.squeeze(0).cpu().float().clamp(-1, 1)
    arr = ((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(arr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Extraire I0 et I1 du GIF ---
gif = Image.open(GIF_PATH)
frames = []
try:
    while True:
        frames.append(gif.copy().convert("RGB"))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

print(f"GIF: {len(frames)} frames")
I0_pil = frames[0]
I1_pil = frames[-1]

I0_pil.save(os.path.join(OUT_DIR, "I0.png"))
I1_pil.save(os.path.join(OUT_DIR, "I1.png"))
print(f"Saved I0.png and I1.png")

# --- Charger modèles ---
print("Loading VAE...")
vae = FrozenVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)

print(f"Loading DiffusionUNetV3 from {CKPT}...")
ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
model = DiffusionUNetV3(latent_ch=4, base_ch=BASE_CH).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Epoch {ckpt['epoch']} | best_psnr={ckpt.get('best_psnr', 0):.3f}")

schedule = CosineNoiseSchedule(T=T).to(device)
sampler  = DDIMSampler(model, schedule, T_inf=T_INF, eta=0.0, prediction_type="v")

# --- Inférence ---
I0 = pil_to_tensor(I0_pil).to(device)
I1 = pil_to_tensor(I1_pil).to(device)

print(f"Running DDIM ({T_INF} steps)...")
with torch.no_grad():
    z0     = vae.encode(I0)
    z1     = vae.encode(I1)
    z_pred = sampler.sample(z0, z1, device=device)
    pred   = vae.decode(z_pred)

pred_pil = tensor_to_pil(pred)
pred_pil.save(os.path.join(OUT_DIR, "Imid_diffusion.png"))
print(f"Saved Imid_diffusion.png")

# --- GIF final : I0 → Imid_diffusion → I1 ---
gif_out = os.path.join(OUT_DIR, "result.gif")
I0_pil.resize((256, 256), Image.LANCZOS).save(
    gif_out, save_all=True,
    append_images=[pred_pil, I1_pil.resize((256, 256), Image.LANCZOS)],
    duration=500, loop=0
)
print(f"Saved result.gif  →  {OUT_DIR}/")
print("Order: I0 → Imid_diffusion → I1")
