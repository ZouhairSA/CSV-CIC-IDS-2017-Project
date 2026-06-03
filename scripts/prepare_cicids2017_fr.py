"""
prepare_cicids2017_fr.py

Charge plusieurs fichiers CSV CIC-IDS-2017 depuis DATASET_PATH, les fusionne,
nettoie le jeu de données et renvoie un DataFrame prêt pour apprentissage
(étiquette binaire : Normal=0, Attaque=1).

Dépendances : pandas, numpy
Définir DATASET_PATH sur le dossier contenant les fichiers CSV.
"""

import os
import glob
from typing import Tuple
import numpy as np
import pandas as pd

# Chemin vers le dossier contenant les CSV (modifier si nécessaire)
DATASET_PATH = r"c:\Users\sabyo\Desktop\CIC-IDS- 2017-Project\data\raw"

# Paramètres
MISSING_COL_THRESHOLD = 0.5  # supprimer les colonnes avec >50% de valeurs manquantes
VERBOSE = True


def load_csv_files(path: str) -> pd.DataFrame:
    """Charger tous les CSV du dossier et concaténer en un seul DataFrame."""
    pattern = os.path.join(path, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {path!r}")

    dfs = []
    for f in files:
        if VERBOSE:
            print(f"Chargement de {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    return merged


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiser les noms de colonnes : suppression d'espaces, minuscules, caractères spéciaux."""
    def _clean(col: str) -> str:
        col = str(col).strip()
        col = col.replace(" ", "_").replace("-", "_").replace("/", "_")
        col = col.replace("(", "").replace(")", "").replace("%", "pct")
        col = col.lower()
        col = "_".join([p for p in col.split("_") if p != ""])    
        return col

    new_cols = {c: _clean(c) for c in df.columns}
    df = df.rename(columns=new_cols)
    return df


def replace_infinite_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Remplacer inf / -inf par NaN pour les colonnes numériques."""
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].mask(~np.isfinite(df[numeric]))
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold: float = MISSING_COL_THRESHOLD) -> pd.DataFrame:
    """Supprimer les colonnes ayant une fraction de valeurs manquantes > threshold."""
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if to_drop and VERBOSE:
        print(f"Suppression de {len(to_drop)} colonnes avec >{threshold*100:.0f}% de valeurs manquantes")
    return df.drop(columns=to_drop) if to_drop else df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputer les valeurs manquantes :
    - numériques : médiane
    - object/strings : mode (valeur la plus fréquente) ou 'missing'
    Stratégie robuste pour ML.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
            df[col] = df[col].fillna(fill)

    cat_cols = df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                fill = mode.iloc[0]
            else:
                fill = df[col].cat.categories[0] if df[col].cat.categories.size else -1
            df[col] = df[col].fillna(fill)

    return df


def encode_label_binary(df: pd.DataFrame, target_candidates=("label", "class")) -> Tuple[pd.DataFrame, str]:
    """
    Trouver la colonne cible et encoder en binaire :
    - Normal / Benign -> 0
    - tout le reste -> 1
    Retourne (df, nom_colonne_originale)
    """
    cols_lower = {c.lower(): c for c in df.columns}
    found = None
    for cand in target_candidates:
        if cand in cols_lower:
            found = cols_lower[cand]
            break
    if found is None:
        for k, v in cols_lower.items():
            if "label" in k or "class" in k or "attack" in k:
                found = v
                break
    if found is None:
        raise KeyError("Impossible de trouver une colonne d'étiquette (ex. 'Label').")

    col = found
    values = df[col].astype(str).str.strip().str.lower()
    is_normal = values.isin({"benign", "normal", "normal_traffic", "none"})
    df["target"] = np.where(is_normal, 0, 1).astype(np.uint8)
    if VERBOSE:
        print(f"Colonne d'étiquette '{col}' encodée en binaire -> 'target' (0 normal, 1 attaque)")
    return df, col


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Réduire la taille mémoire : downcast numérique et convertir objets peu cardinalité en 'category'."""
    int_cols = df.select_dtypes(include=["int64", "int32"]).columns
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns

    for col in int_cols:
        try:
            df[col] = pd.to_numeric(df[col], downcast="unsigned" if (df[col] >= 0).all() else "integer")
        except Exception:
            pass

    for col in float_cols:
        try:
            df[col] = pd.to_numeric(df[col], downcast="float")
        except Exception:
            pass

    obj_cols = df.select_dtypes(include=["object"]).columns
    n_total = len(df)
    for col in obj_cols:
        n_unique = df[col].nunique(dropna=False)
        if n_total > 0 and (n_unique / n_total) < 0.5:
            df[col] = df[col].astype("category")

    return df


def print_dataset_info(stage: str, df: pd.DataFrame):
    """Afficher forme, nombre de classes et distribution si 'target' présent."""
    print(f"\n[{stage}] forme : {df.shape}")
    if "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        n_classes = counts.shape[0]
        print(f"[{stage}] nombre de classes (target) : {n_classes}")
        print(f"[{stage}] distribution des classes (comptes) :\n{counts.to_dict()}")
        print(f"[{stage}] distribution des classes (%) :\n{(counts / counts.sum() * 100).round(3).to_dict()}")
    else:
        print(f"[{stage}] colonne 'target' absente")


def prepare_dataset(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Pipeline principal : charger -> fusionner -> nettoyer -> encoder -> optimiser -> retourner df."""
    df = load_csv_files(dataset_path)
    print_dataset_info("après_fusion", df)

    df = standardize_column_names(df)
    df = replace_infinite_with_nan(df)
    df = drop_sparse_columns(df, threshold=MISSING_COL_THRESHOLD)
    print_dataset_info("avant_nettoyage", df)

    df = impute_missing_values(df)
    df, original_label_col = encode_label_binary(df)
    print_dataset_info("après_nettoyage", df)

    df = downcast_dtypes(df)
    if VERBOSE:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"\nMémoire utilisée après optimisation : {mem_mb:.2f} MB")

    return df


if __name__ == "__main__":
    final_df = prepare_dataset(DATASET_PATH)
    print("\nDataFrame final prêt pour ML. Exemples de colonnes :", final_df.columns[:10].tolist())
    # Sauvegarde automatique du DataFrame nettoyé dans le dossier results/ à la racine du projet
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "cicids2017_nettoye.parquet")
    try:
        final_df.to_parquet(out_path, index=False)
        if VERBOSE:
            print(f"DataFrame nettoyé sauvegardé : {out_path}")
    except Exception as e:
        print(f"Échec de la sauvegarde en parquet : {e}. Tentative de sauvegarde en CSV compressé...")
        # fallback : sauvegarder en CSV compressé si parquet non disponible
        out_path_csv = os.path.join(results_dir, "cicids2017_nettoye.csv.gz")
        try:
            final_df.to_csv(out_path_csv, index=False, compression="gzip")
            if VERBOSE:
                print(f"DataFrame nettoyé sauvegardé en CSV compressé : {out_path_csv}")
        except Exception as e2:
            print(f"Échec de la sauvegarde en CSV compressé : {e2}. Sauvegarde manuelle requise.")
