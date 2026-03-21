"""
poster/12_gen_motion_cues.py — Génère |I1-I0| et 0.5*(I0+I1) depuis I0.png / I1.png
Sauvegarde dans le même dossier que I0.png.
"""
import os
import numpy as np
from PIL import Image

FOLDER = os.path.join(os.path.dirname(__file__), "results", "of_example")

I0 = np.array(Image.open(os.path.join(FOLDER, "I0.png"))).astype(np.float32)
I1 = np.array(Image.open(os.path.join(FOLDER, "I1.png"))).astype(np.float32)

# |I1 - I0| — amplify x4 pour la visibilité (sinon trop sombre)
abs_diff = np.abs(I1 - I0)
abs_diff_vis = np.clip(abs_diff * 4, 0, 255).astype(np.uint8)
Image.fromarray(abs_diff_vis).save(os.path.join(FOLDER, "abs_diff.png"))
print(f"Saved: {FOLDER}/abs_diff.png")

# 0.5*(I0+I1)
avg = np.clip((I0 + I1) / 2.0, 0, 255).astype(np.uint8)
Image.fromarray(avg).save(os.path.join(FOLDER, "avg.png"))
print(f"Saved: {FOLDER}/avg.png")
