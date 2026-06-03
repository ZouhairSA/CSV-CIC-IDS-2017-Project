"""
Génère des sous-ensembles (subsets) échantillonnés depuis
`results/cicids2017_nettoye.parquet` (ou le fallback CSV gz).

Usage (PowerShell):
& .\cicids_env\Scripts\Activate.ps1
python .\scripts\generate_subsets_fr.py --sizes 1000,5000,10000,20000

Le script essaie de stratifier l'échantillonnage sur la colonne `target`
si elle est présente. Les fichiers sont écrits dans `results/` au format
Parquet si possible, sinon CSV gz.
"""
import os
import argparse
import logging
import sys
from typing import List

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_sizes(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    sizes = []
    for p in parts:
        try:
            sizes.append(int(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Taille invalide: {p}")
    return sizes


def read_cleaned(results_dir: str) -> pd.DataFrame:
    p_parquet = os.path.join(results_dir, 'cicids2017_nettoye.parquet')
    p_csv = os.path.join(results_dir, 'cicids2017_nettoye.csv.gz')
    if os.path.exists(p_parquet):
        logging.info(f"Lecture de {p_parquet}")
        return pd.read_parquet(p_parquet)
    if os.path.exists(p_csv):
        logging.info(f"Lecture de {p_csv}")
        return pd.read_csv(p_csv, compression='gzip')
    raise FileNotFoundError('Aucun fichier nettoyé trouvé dans results/ (parquet ou csv.gz)')


def stratified_sample(df: pd.DataFrame, n: int, target_col: str = 'target', random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    vc = df[target_col].value_counts()
    # proportions
    props = (vc / vc.sum()).to_dict()
    # initial allocation
    alloc = {k: max(1, int(round(props[k] * n))) for k in props}
    # adjust to match n
    total = sum(alloc.values())
    if total != n:
        # sort classes by count desc and adjust
        keys_sorted = sorted(alloc.keys(), key=lambda k: vc[k], reverse=True)
        i = 0
        while total < n:
            alloc[keys_sorted[i % len(keys_sorted)]] += 1
            total += 1
            i += 1
        i = 0
        while total > n:
            if alloc[keys_sorted[i % len(keys_sorted)]] > 1:
                alloc[keys_sorted[i % len(keys_sorted)]] -= 1
                total -= 1
            i += 1
    parts = []
    for cls, k in alloc.items():
        sub = df[df[target_col] == cls]
        if k >= len(sub):
            logging.warning(f"Classe {cls}: demande {k} >= disponible {len(sub)} -> utiliser tout")
            parts.append(sub.sample(n=len(sub), random_state=random_state))
        else:
            parts.append(sub.sample(n=k, random_state=random_state))
    res = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return res


def save_df(df: pd.DataFrame, out_path_noext: str) -> str:
    # out_path_noext is full path without extension
    p_parquet = out_path_noext + '.parquet'
    p_csvgz = out_path_noext + '.csv.gz'
    try:
        df.to_parquet(p_parquet, index=False)
        return p_parquet
    except Exception as e:
        logging.warning(f"Écriture Parquet impossible ({e}), fallback CSV gz")
        df.to_csv(p_csvgz, index=False, compression='gzip')
        return p_csvgz


def main():
    parser = argparse.ArgumentParser(description='Génère des subsets depuis le DataFrame nettoyé CIC-IDS-2017')
    parser.add_argument('--sizes', type=parse_sizes, default=[1000,5000,10000,20000,50000,100000,200000,500000],
                        help='Liste de tailles séparées par des virgules, ex: 1000,5000,10000')
    parser.add_argument('--seed', type=int, default=42, help='Seed RNG')
    parser.add_argument('--results-dir', default=None, help='Chemin vers le dossier results (détecté automatiquement)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.normpath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    try:
        df = read_cleaned(results_dir)
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)

    total = len(df)
    logging.info(f"Total en entrée: {total} lignes, colonnes: {len(df.columns)}")

    has_target = 'target' in df.columns
    if not has_target:
        logging.warning("La colonne 'target' n'existe pas -> échantillonnage non-stratifié")

    for n in args.sizes:
        if n <= 0:
            logging.info(f"Taille ignorée (<=0): {n}")
            continue
        if n > total:
            logging.info(f"Taille {n} > total {total} — sauté")
            continue
        logging.info(f"Génération subset n={n}")
        if has_target:
            samp = stratified_sample(df, n, target_col='target', random_state=args.seed)
        else:
            samp = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        out_noext = os.path.join(results_dir, f'subset_{n}')
        written = save_df(samp, out_noext)
        logging.info(f"Écrit {written} (shape={samp.shape})")


if __name__ == '__main__':
    main()
