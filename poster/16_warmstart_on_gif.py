"""
poster/16_warmstart_on_gif.py — Warm start diffusion sur ep005_s03.gif

Pipeline:
    I0, I1  <- frames du GIF
    Imid_pixel = UNetPixelV2(I0, I1)          # prediction grossiere
    z_pixel    = VAE.encode(Imid_pixel)
    z_noisy    = q_sample(z_pixel, t=T_START)  # bruit partiel
    z_denoised = DDIM(z_noisy, T_START -> 0)  # raffinage
    Imid_final = VAE.decode(z_denoised)

Sorties dans results/warmstart_on_gan/:
    I0.png, I1.png, Imid_pixel.png, Imid_noisy.png, Imid_final.png
"""
import os, sys, torch
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.unet_pixel_v2 import UNetPixelV2
from models.unet_latent   import FrozenVAE
from models.diffusion_v3  import DiffusionUNetV3, CosineNoiseSchedule

GIF_PATH   = os.path.join(ROOT, "checkpoints/gan_scratch/samples/ep005_s03.gif")
PIXEL_CKPT = os.path.join(ROOT, "checkpoints/pixel_v2/best.pth")
DIFF_CKPT  = os.path.join(ROOT, "checkpoints/diffusion_v3/best.pth")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results/warmstart_on_gan")
os.makedirs(OUT_DIR, exist_ok=True)

T        = 1000
T_START  = 200   # niveau de bruit ajouté
T_INF    = 50    # steps DDIM pour débruiter
BASE_CH  = 128


def pil_to_tensor(img):
    img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def tensor_to_pil(t):
    t = t.squeeze(0).cpu().float().clamp(-1, 1)
    return Image.fromarray(((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy())


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
I0 = pil_to_tensor(I0_pil).to(device)
I1 = pil_to_tensor(I1_pil).to(device)

tensor_to_pil(I0).save(os.path.join(OUT_DIR, "I0.png"))
tensor_to_pil(I1).save(os.path.join(OUT_DIR, "I1.png"))

# --- UNetPixelV2 : prediction grossiere ---
print(f"Loading UNetPixelV2 from {PIXEL_CKPT}...")
px_ckpt = torch.load(PIXEL_CKPT, map_location=device, weights_only=False)
px_args = px_ckpt.get("args", {})
pixel_model = UNetPixelV2(
    base_ch      = px_args.get("base_ch", 64),
    use_residual = not px_args.get("no_residual", False),
    use_refine   = not px_args.get("no_refine", False),
).to(device)
pixel_model.load_state_dict(px_ckpt["model"])
pixel_model.eval()
print(f"  Epoch {px_ckpt.get('epoch', '?')} | best_psnr={px_ckpt.get('best_psnr', 0):.3f}")

with torch.no_grad():
    Imid_pixel = pixel_model(I0, I1).clamp(-1, 1)

tensor_to_pil(Imid_pixel).save(os.path.join(OUT_DIR, "Imid_pixel.png"))
print("Saved Imid_pixel.png")

# --- VAE ---
print("Loading VAE...")
vae = FrozenVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)

with torch.no_grad():
    z0      = vae.encode(I0)
    z1      = vae.encode(I1)
    z_pixel = vae.encode(Imid_pixel)

# --- Bruiter z_pixel au timestep T_START ---
schedule = CosineNoiseSchedule(T=T).to(device)
t_tensor = torch.tensor([T_START], device=device, dtype=torch.long)
with torch.no_grad():
    z_noisy, _ = schedule.q_sample(z_pixel, t_tensor)
    Imid_noisy = vae.decode(z_noisy)

tensor_to_pil(Imid_noisy).save(os.path.join(OUT_DIR, "Imid_noisy.png"))
print(f"Saved Imid_noisy.png  (t={T_START})")

# --- DiffusionUNetV3 : DDIM warm start ---
print(f"Loading DiffusionUNetV3 from {DIFF_CKPT}...")
d_ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=False)
diff_model = DiffusionUNetV3(latent_ch=4, base_ch=BASE_CH).to(device)
diff_model.load_state_dict(d_ckpt["model"])
diff_model.eval()
print(f"  Epoch {d_ckpt['epoch']} | best_psnr={d_ckpt.get('best_psnr', 0):.3f}")

# DDIM depuis t=T_START → 0 (v-prediction)
ab = schedule["alpha_bar"]
timesteps = torch.linspace(T_START, 1, T_INF, dtype=torch.long).to(device)
ts_prev   = torch.cat([timesteps[1:], torch.zeros(1, dtype=torch.long, device=device)])

z = z_noisy.clone()
print(f"Running DDIM warm start ({T_INF} steps, t={T_START}->0)...")
with torch.no_grad():
    for t_cur, t_prev in zip(timesteps, ts_prev):
        t_batch = torch.full((1,), t_cur.item(), device=device, dtype=torch.long)
        out     = diff_model(z, t_batch, z0, z1)

        a_cur  = ab[t_cur].to(device)
        a_prev = ab[t_prev].to(device) if t_prev > 0 else torch.tensor(1.0, device=device)

        # v-prediction: x0 = sqrt(a)*z - sqrt(1-a)*v
        x0_pred = a_cur.sqrt() * z - (1 - a_cur).sqrt() * out
        x0_pred = x0_pred.clamp(-10, 10)
        eps_eq  = (1 - a_cur).sqrt() * z + a_cur.sqrt() * out
        dir_xt  = (1 - a_prev).clamp(0).sqrt() * eps_eq
        z       = a_prev.sqrt() * x0_pred + dir_xt

    Imid_final = vae.decode(z)

tensor_to_pil(Imid_final).save(os.path.join(OUT_DIR, "Imid_final.png"))
print("Saved Imid_final.png")

# --- GIF final ---
gif_out = os.path.join(OUT_DIR, "result.gif")
tensor_to_pil(I0).save(
    gif_out, save_all=True,
    append_images=[
        tensor_to_pil(Imid_pixel),
        tensor_to_pil(Imid_noisy),
        tensor_to_pil(Imid_final),
        tensor_to_pil(I1),
    ],
    duration=600, loop=0
)
print(f"Saved result.gif  (I0 -> pixel -> noisy -> final -> I1)")
print(f"\nAll outputs in: {OUT_DIR}")
