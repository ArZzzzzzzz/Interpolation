"""
poster/run_all.py — Lance tous les scripts de génération dans l'ordre

Usage:
    cd poster
    python run_all.py --data_root "D:/projet mumltimodal/vimeo_triplet"
    python run_all.py --data_root "D:/projet mumltimodal/vimeo_triplet" --skip_diffusion
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, extra_args=""):
    cmd = f"{sys.executable} {os.path.join(HERE, script)} {extra_args}"
    print(f"\n{'='*60}")
    print(f"  Lancement : {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=HERE)
    if result.returncode != 0:
        print(f"[!] {script} a échoué (code {result.returncode})")
    return result.returncode == 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      type=str, default="D:/projet mumltimodal/vimeo_triplet")
    p.add_argument("--video",          type=str, default=os.path.join(ROOT, "19201080_before.mp4"))
    p.add_argument("--skip_diffusion", action="store_true", help="Sauter la diffusion (inference lente)")
    p.add_argument("--n_batches_mad",  type=int, default=200)
    p.add_argument("--n_eval",         type=int, default=100)
    p.add_argument("--n_samples",      type=int, default=6)
    p.add_argument("--T_inf",          type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    # 1. MAD vs quality
    run("01_mad_vs_quality.py",
        f'--data_root "{args.data_root}" --n_batches {args.n_batches_mad}')

    # 2. Tile demo
    run("02_tile_demo.py",
        f'--video "{args.video}"')

    # 3. Model comparison
    skip = "--skip_diffusion" if args.skip_diffusion else ""
    run("03_model_comparison.py",
        f'--data_root "{args.data_root}" --n_eval {args.n_eval} --n_samples {args.n_samples} --T_inf {args.T_inf} {skip}')

    # 4. Tile visual (schéma pédagogique, rapide)
    run("04_tile_visual.py",
        f'--video "{args.video}"')

    print(f"\n{'='*60}")
    print("  Terminé ! Résultats dans poster/results/")
    print(f"{'='*60}")
    for f in sorted(os.listdir(os.path.join(HERE, "results"))):
        print(f"    {f}")


if __name__ == "__main__":
    main()
