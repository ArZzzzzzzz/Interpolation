# Poster — Video Frame Interpolation (VFI)

## Scripts de génération

| Script | Génère | Durée |
|---|---|---|
| `01_mad_vs_quality.py` | `results/mad_vs_quality.png` | ~5 min |
| `02_tile_demo.py` | `results/tile_demo.png`, `tile_comparison.png`, `tile_diff.png` | ~3 min |
| `03_model_comparison.py` | `results/comparison_XXX.png`, `metrics_table.png`, `psnr_bar.png` | ~15 min (sans diffusion) |
| `04_tile_visual.py` | `results/tile_grid_overview.png`, `tile_overlap_zoom.png`, `tile_schema_1d.png` | ~30 sec |
| `run_all.py` | Lance tout | ~25 min |

```bash
cd poster
python run_all.py --data_root "D:/projet mumltimodal/vimeo_triplet" --skip_diffusion
# ou tout avec diffusion :
python run_all.py --data_root "D:/projet mumltimodal/vimeo_triplet"
```

---

## Métriques connues (Vimeo-90K val)

| Modèle | PSNR (dB) | SSIM | LPIPS | Note |
|---|---|---|---|---|
| Moyenne (baseline) | 24.64 | 0.7404 | 0.0017 | (I0+I1)/2 |
| **UNetPixelV2** | **33.26** | **0.9515** | **0.0005** | 100 batches val |
| UNetPixelV2 + GAN | 32.31 | 0.9432 | 0.0006 | 100 batches val, Vimeo |
| DiffusionUNetV3 | 26.57 | — | — | DDIM 50 steps val (checkpoint val) |

---

## État de l'art — VFI

### Problème
Générer Imid à partir de (I0, I1) : modéliser le mouvement inter-frames est difficile car :
- Occultations (zones visibles dans I1 mais pas I0)
- Mouvements non-rigides
- Grands déplacements (flou de mouvement)

### Approches classiques (basées sur le flot optique)
- **DAIN** (Bao et al., 2019) : depth-aware optical flow + warping
- **RIFE** (Huang et al., 2022) : intermediate flow estimation, temps réel
- **FILM** (Reda et al., 2022) : feature pyramids multi-scale, robuste aux grands mouvements

### Approches par apprentissage direct (sans flot explicite)
- **SuperSloMo** (Jiang et al., 2018) : U-Net + optical flow estimation conjointe
- **ABME** (Park et al., 2021) : asymmetric bilateral motion estimation
- **IFRNet** (Kong et al., 2022) : coarse-to-fine feature refinement

### Approches en espace latent / génératif
- **LDMVFI** (Danier et al., 2023) : Latent Diffusion pour VFI
- **MCVD** (Voleti et al., 2022) : masked conditional video diffusion
- **ToonCrafter** (2024) : VFI conditionnel pour animation, basé sur SD

### Notre position
Nous comparons 3 familles d'approches sur le même dataset (Vimeo-90K) :
1. **Pixel direct** (UNetPixelV2) — rapide, haute fidélité
2. **Diffusion conditionnelle** (DiffusionUNetV3) — plus de détails, plus lent
3. **Affinage par GAN** (UNetPixelV2 + PatchGAN) — sharpness supplémentaire

---

## Architectures des modèles (pour les posters)

### 1. UNetPixelV2
```
I0 [3ch] ─┐
I1 [3ch] ─┼→ [I0, I1, I1-I0, |I1-I0|] (12ch)
           │
           ▼ Encoder (4 niveaux, MaxPool)
     ResBlock (64ch)  → ResBlock (128ch) → ResBlock (256ch) → ResBlock (512ch)
     [GroupNorm + SiLU, skip connections]
           ▼ Bottleneck (2× ResBlock 512ch)
     Decoder (4 niveaux, Bilinear upsample + skip cat)
     ResBlock (256ch) → ResBlock (128ch) → ResBlock (64ch)
           ▼
     Conv 1×1 → résidu sur 0.5*(I0+I1)     [init poids à 0]
           ▼
     RefinementNet : [I0, I1, coarse] → correction haute fréquence (4 convolutions)
           ▼ Imid_pred [3ch]
```
- **~15M paramètres** | crop 256×256 | loss Charbonnier + Perceptual + Gradient + Motion-weighted
- Inférence HD via découpage en tuiles avec overlap (voir Script 02/04)

### 2. UNetLatent (VAE latent space)
```
I0 ──→ VAE Encoder (gelé) ──→ z0 [4ch, H/8, W/8]
I1 ──→ VAE Encoder (gelé) ──→ z1 [4ch, H/8, W/8]
                cat(z0, z1) = 8ch
                      ↓
              UNetLatent (4 niveaux, GN+SiLU)
                      ↓ z_pred [4ch]
              VAE Decoder (gelé) → Imid_pred [3ch]
```
- VAE : `stabilityai/sd-vae-ft-mse`, scale_factor=0.18215, downscale 8×
- Travail dans l'espace sémantique compressé (~4× moins de pixels)

### 3. DiffusionUNetV3 (ControlNet-style)
```
z0, z1 ──→ CondEncoder (4 niveaux) ──→ [cond0, cond1, cond2, cond_mid]
                                          ↓ injection additive à chaque niveau

z_T (bruit pur) ──→ Débruiteur UNet (z_t + timestep AdaGN)
                         ↓ + cond_X à chaque niveau du décodeur
                    eps_pred / v_pred
                         ↓ DDIM 250 steps (eta=0)
                    z_pred → VAE Decode → Imid_pred
```
- Avantage vs V1/V2 : le conditionnement I0/I1 est présent à **chaque résolution**
- Training : T=1000, v-prediction, cosine schedule, ~48M paramètres

### 4. UNetPixelV2 + GAN (fine-tuning phase 2)
```
Générateur G = UNetPixelV2  [initialisé depuis pixel_v2/best.pth]
Discriminateur D = PatchGAN 70×70  [entraîné depuis zéro]
    Input D : cat(I0, I1, Imid) = 9 canaux
    Output D : patch scores [B, 1, 30, 30]

L_G = λ_charb · Charbonnier(pred, GT)
    + λ_perc  · Perceptual(pred, GT)       [VGG16]
    + λ_gan   · LSGAN_G(D(I0, I1, pred))  [λ_gan=0.02]

L_D = LSGAN_D(D(I0,I1,real), D(I0,I1,fake.detach()))
```
- LSGAN (Mao et al., 2017) : plus stable que BCE
- λ_gan petit (0.02) pour éviter mode collapse
- Fine-tune avec lr_G=5e-5 (vs 2e-4 pour phase 1)

---

## Contenu des 2 Posters

---

### POSTER 1 — Introduction, bibliographie et méthodologie

#### Introduction
**Tâche** : Video Frame Interpolation (VFI) — générer la frame temporellement centrale Imid
entre deux frames I0 et I1.

**Application principale** : conversion 25fps → 50fps pour vidéos HD (1920×1080).
Dataset d'entraînement : Vimeo-90K Triplet (91 701 triplets, 256×256).

**Défi clé** : quand le mouvement inter-frames est grand (MAD élevé),
un modèle pixel prédit une "moyenne floue" et le PSNR chute.
→ **Notre analyse MAD vs PSNR** (Script 01) quantifie ce seuil (~MAD=15 en [0,255]).

#### Bibliographie (sélective)
- Jiang et al. (2018) — SuperSloMo : U-Net + optical flow pour slow-motion
- Huang et al. (2022) — RIFE : estimation de flot intermédiaire en temps réel
- Ho et al. (2020) — DDPM : bases du modèle de diffusion débruisant
- Rombach et al. (2022) — LDM / Stable Diffusion : diffusion en espace latent
- Isola et al. (2017) — pix2pix / PatchGAN : discriminateur conditionnel 70×70
- Mao et al. (2017) — LSGAN : GAN moindres carrés, plus stable
- Danier et al. (2023) — LDMVFI : premier VFI par diffusion latente
- Johnson et al. (2016) — Perceptual losses (VGG features)

#### Démarche / Pipeline

```
Étape 1 : UNetPixelV2 (pixel direct, 256×256)
  → PSNR=33.63 dB | Problème : flou sur grand mouvement

Étape 2 : Inférence tiled pour vidéos HD (1920×1080)
  → découpage en tuiles 256×256 avec overlap=32px, moyennage des zones partagées
  → Script 04 : visualisation de la grille + zones d'overlap

Étape 3 : Analyse du seuil de dégradation (MAD vs PSNR)
  → Script 01 : courbe PSNR = f(MAD), identification du seuil

Étape 4a : DiffusionUNetV3 — récupérer le détail via génération stochastique
  → DDIM 250 steps depuis bruit pur, conditionné sur (z0, z1) à chaque échelle

Étape 4b : UNetPixelV2 + GAN — récupérer le détail via adversarial training
  → PatchGAN 70×70 + LSGAN, λ_gan=0.02 sur UNetPixelV2 pré-entraîné
```

---

### POSTER 2 — Détails, résultats, cas d'échec

#### Détails supplémentaires

**Normalisation & preprocessing**
- Images normalisées dans [-1, 1]
- Augmentation temporelle cohérente : affine (rotation ±5°, translation ±5%) appliqué identiquement sur I0/Imid/I1

**Loss UNetPixelV2 (VFILossV2)**
```
L = 1.0 · Charbonnier(pred, GT)
  + 0.1 · Perceptual(pred, GT)    [VGG16 relu2_2 + relu3_3]
  + 0.1 · Gradient(pred, GT)      [Sobel edges]
  + 0.1 · MotionWeighted(pred, GT) [pondéré par |I1-I0|]
```
Appliqué à 3 résolutions : full (×1.0), half (×0.5), quarter (×0.25).

**Paramètres d'entraînement**
- AdamW, lr=2e-4, cosine decay, weight_decay=1e-4
- Mixed precision (fp16), gradient clipping max_norm=1.0
- 20 epochs (pixel_v2) | 30 epochs GAN | 45 epochs diffusion_v3

**Inférence tiled HD**
- patch_size=256, overlap=32 → stride=224
- Grille : ~9×5 = 45 tuiles pour 1920×1080
- Moyennage pondéré sur les zones d'overlap (poids uniforme)

#### Résultats quantitatifs

| Modèle | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | Inference |
|---|---|---|---|---|
| Moyenne (baseline) | 24.64 | 0.7404 | 0.0017 | <1ms |
| **UNetPixelV2** | **33.26** | **0.9515** | **0.0005** | ~10ms/frame |
| UNetPixelV2 + GAN | 32.31 | 0.9432 | 0.0006 | ~10ms/frame |
| DiffusionUNetV3 | 26.57 | — | — | ~8s/frame (50 steps) |

**Note** : LPIPS capture mieux la qualité perçue que le PSNR seul.
Le modèle de diffusion peut avoir un LPIPS meilleur malgré un PSNR plus bas.

#### Résultats qualitatifs
→ Figures `comparison_XXX.png` : I0 | I1 | GT | Moyenne | PixelV2 | GAN | Diffusion
→ Figures `mad_vs_quality.png` : PSNR = f(MAD)
→ Figures `tile_overlap_zoom.png` + `tile_schema_1d.png` : stratégie tiling

#### Cas d'échec / observations intéressantes

**UNetPixelV2**
- ✗ Flou résiduel sur les zones de fort mouvement (bords, sports)
- ✗ Artefacts de stitching sur les bords des tuiles (atténués par overlap)
- ✓ Rapide, stable, très bon PSNR

**GAN (pixel_v2_gan)**
- ✗ PSNR plus bas (GAN ne maximise pas la fidélité pixel-à-pixel)
- ✓ Texture plus nette, détails récupérés (perçu meilleur à l'œil)
- ✗ Risque de mode collapse si λ_gan trop élevé
- Note : entraîné sur dataset vidéo différent → résultats non directement comparables

**DiffusionUNetV3**
- ✗ PSNR plus bas (diversité ≠ fidélité)
- ✓ Meilleure qualité texturale perçue (LPIPS potentiellement meilleur)
- ✗ Très lent : 250 passes DDIM par frame (~8s sur GPU)
- ✓ Le conditionnement ControlNet-style (injection à chaque échelle) améliore
      la cohérence spatiale vs V1/V2 (injection unique à l'entrée)

**Observation MAD**
- Au-delà de MAD≈15, l'écart UNetPixelV2 vs Baseline se réduit
- Cela motive l'utilisation de la diffusion / GAN sur les frames difficiles
- Piste : sélection adaptative du modèle selon le MAD de la paire (I0, I1)
