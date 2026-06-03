"""
Exécute des expériences sur les subsets présents dans `results/`.
- Cherche `results/subset_<N>.parquet` ou `.csv.gz`.
- Pour chaque subset :
  - Charge les données
  - Sélectionne features numériques (exclut `target`)
  - Split train/test stratifié (80/20)
  - Pipelines : imputer(median) -> StandardScaler -> LogisticRegression
  - Enregistre métriques dans `results/experiments_results.csv`

Usage :
& .\cicids_env\Scripts\python.exe .\scripts\run_experiments_fr.py

Remarques :
- Le script saute les fichiers trop petits ou mal formés.
- Nécessite scikit-learn, pandas, numpy.
"""
import os
import glob
import time
import logging
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def find_subsets(results_dir: str) -> List[str]:
    pats = [os.path.join(results_dir, 'subset_*.parquet'), os.path.join(results_dir, 'subset_*.csv.gz')]
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(p)))
    return files


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    if path.lower().endswith('.csv.gz') or path.lower().endswith('.csv'):
        return pd.read_csv(path, compression='gzip' if path.lower().endswith('.csv.gz') else None)
    raise ValueError('Format inconnu: ' + path)


def extract_size(path: str) -> int:
    base = os.path.basename(path)
    # subset_1000.parquet
    parts = base.split('_')
    if len(parts) < 2:
        return -1
    num = parts[1]
    num = os.path.splitext(num)[0]
    try:
        return int(num)
    except Exception:
        # try to parse digits
        import re
        m = re.search(r'(\d+)', base)
        return int(m.group(1)) if m else -1


def compute_metrics(y_true, y_pred, y_score) -> Dict:
    res = {}
    res['accuracy'] = float(accuracy_score(y_true, y_pred))
    res['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    res['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    res['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        res['roc_auc'] = float(roc_auc_score(y_true, y_score))
    except Exception:
        res['roc_auc'] = float('nan')
    return res


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    subset_files = find_subsets(results_dir)
    if not subset_files:
        logging.error('Aucun subset trouvé dans results/ (subset_*.parquet ou subset_*.csv.gz)')
        sys.exit(1)

    logging.info(f'Trouvé {len(subset_files)} subset(s)')
    rows = []

    for path in subset_files:
        logging.info('---')
        logging.info(f'Chargement: {path}')
        try:
            df = load_df(path)
        except Exception as e:
            logging.error(f'Échec lecture {path}: {e}')
            continue
        if 'target' not in df.columns:
            logging.error(f"La colonne 'target' est absente dans {path} — sauté")
            continue
        n_rows = len(df)
        size = extract_size(path)
        if size <= 0:
            size = n_rows
        logging.info(f'Shape: {df.shape} | size (parsed) = {size}')

        # features numeric
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric:
            numeric.remove('target')
        if not numeric:
            logging.error(f'Aucune feature numérique dans {path} — sauté')
            continue
        X = df[numeric]
        y = df['target'].astype(int)

        # train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Prétraitement séparé : imputer + scaler (non mesurés)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        try:
            imputer.fit(X_train)
            X_train_imp = imputer.transform(X_train)
            X_test_imp = imputer.transform(X_test)
            scaler.fit(X_train_imp)
            X_train_pre = scaler.transform(X_train_imp)
            X_test_pre = scaler.transform(X_test_imp)
        except Exception as e:
            logging.error(f'Erreur prétraitement pour {path}: {e}')
            continue

        # Classifieur : mesurer uniquement son fit
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)
        start = time.perf_counter()
        try:
            clf.fit(X_train_pre, y_train)
        except Exception as e:
            logging.error(f'Echec entraînement pour {path}: {e}')
            continue
        train_time = time.perf_counter() - start

        n_iter = getattr(clf, 'n_iter_', None)
        if isinstance(n_iter, (list, tuple, np.ndarray)):
            try:
                n_iter_val = int(np.max(n_iter))
            except Exception:
                n_iter_val = -1
        elif n_iter is None:
            n_iter_val = -1
        else:
            n_iter_val = int(n_iter)

        # prédictions et scores
        try:
            y_pred = clf.predict(X_test_pre)
            if hasattr(clf, 'predict_proba'):
                y_score = clf.predict_proba(X_test_pre)[:, 1]
            else:
                y_score = clf.decision_function(X_test_pre)
        except Exception:
            logging.error(f'Echec prédiction pour {path}')
            continue

        metrics = compute_metrics(y_test, y_pred, y_score)

        row = {
            'dataset_size': int(size),
            'n_variables': int(len(numeric)),
            'train_time_seconds': float(train_time),
            'n_iterations': int(n_iter_val),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        }
        logging.info(f'Resultats: {row}')
        rows.append(row)

        # sauvegarde intermédiaire
        out_csv = os.path.join(results_dir, 'experiments_results.csv')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logging.info(f'Enregistré résultats partiels -> {out_csv}')

    if rows:
        out_csv = os.path.join(results_dir, 'experiments_results.csv')
        pd.DataFrame(rows).sort_values('dataset_size').to_csv(out_csv, index=False)
        # tentative d'écriture Excel
        try:
            out_xlsx = os.path.join(results_dir, 'experiments_results.xlsx')
            pd.DataFrame(rows).sort_values('dataset_size').to_excel(out_xlsx, index=False)
            logging.info(f'Excel écrit -> {out_xlsx}')
        except Exception:
            logging.warning('Impossible d\'écrire le XLSX (package manquant?)')
        logging.info(f'Tous résultats écrits -> {out_csv}')
    else:
        logging.warning('Aucun résultat calculé')


if __name__ == '__main__':
    main()
