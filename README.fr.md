# README - Projet CIC-IDS-2017 (français)

## But du dépôt
Préparer et nettoyer les fichiers CSV du dataset CIC-IDS-2017 afin d'obtenir
un DataFrame prêt pour apprentissage (étiquette binaire : Normal=0, Attaque=1),
et fournir un protocole reproductible pour des expérimentations sur sous-ensembles
de tailles variées.

## 1. Organisation et téléchargement des données
- Placez tous les CSV originaux CIC-IDS-2017 dans :
  `data/raw/`
- Le script principal lit tous les CSV présents dans ce dossier.

## 2. Script principal (préparation et nettoyage)
- Fichier : `scripts/prepare_cicids2017_fr.py`
- Actions réalisées automatiquement :
  - Lecture et concaténation de tous les CSV (`data/raw/`)
  - Standardisation des noms de colonnes
  - Remplacement des valeurs infinies par NaN
  - Suppression des colonnes très creuses (>50% manquantes par défaut)
  - Imputation : médiane pour numérique, mode pour catégoriel
  - Encodage binaire de la cible (`target`) : Normal/Benign → 0, autres → 1
  - Downcast des types pour optimiser mémoire
  - Sauvegarde du DataFrame nettoyé dans `results/cicids2017_nettoye.parquet`
    (si `pyarrow` installé) sinon `results/cicids2017_nettoye.csv.gz`.

Exécution (PowerShell) :
```
& .\cicids_env\Scripts\Activate.ps1
python .\scripts\prepare_cicids2017_fr.py
```

## 3. Transformation en classification binaire
- La colonne `target` est ajoutée par `encode_label_binary` dans le script principal.
- Vérifiez les valeurs :
```
python - <<'PY'
import pandas as pd
try:
    df = pd.read_parquet('results/cicids2017_nettoye.parquet')
except Exception:
    df = pd.read_csv('results/cicids2017_nettoye.csv.gz', compression='gzip')
print(df['target'].value_counts())
PY
```

## 4. Préparation des variables
- Le script fait une imputation et convertit automatiquement certaines colonnes
  en `category` si cardinalité faible.
- Pour les expérimentations, on conservera uniquement les colonnes numériques
  (sauf identifiants inutiles) — cf. section 'Génération de sous-ensembles'.

## 5. Génération de sous-ensembles de données (script recommandé)
Objectif : créer des fichiers sous-datasets de tailles différentes pour tester scalabilité.
Taille souhaitées : 1 000, 5 000, 10 000, 20 000, 50 000, 100 000, 200 000, 500 000 (si possible)

Un script prêt-à-lancer a été ajouté : `scripts/generate_subsets_fr.py`.
Résumé :
- Lit `results/cicids2017_nettoye.parquet` ou `results/cicids2017_nettoye.csv.gz`.
- Échantillonne stratifié sur `target` (si présent) pour préserver la proportion Normal/Attaque.
- Sauvegarde chaque subset dans `results/` sous la forme `subset_<N>.parquet` (fallback `subset_<N>.csv.gz`)

Exécution (PowerShell) :
```
& .\cicids_env\Scripts\Activate.ps1
python .\scripts\generate_subsets_fr.py --sizes 1000,5000,10000,20000,50000,100000,200000,500000
```

Notes :
- Si la taille demandée est supérieure au nombre total d'observations, le subset est sauté.
- Vous pouvez ajuster `--sizes` et `--seed` en arguments.

## C. Résultats expérimentaux
Un script `scripts/run_experiments_fr.py` a été ajouté pour exécuter automatiquement les expériences sur tous les fichiers `results/subset_*.parquet` ou `results/subset_*.csv.gz` trouvés.
Pour chaque subset le script :
- Charge les données et conserve uniquement les features numériques (exclut `target`).
- Sépare train/test (80/20) avec stratification sur `target` si possible.
- Exécute le pipeline : Imputer(median) -> StandardScaler -> LogisticRegression(max_iter=1000, class_weight='balanced').
- Mesure et enregistre : temps d'entraînement, nombre d'itérations (model.n_iter_), Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Format de sortie :
- `results/experiments_results.csv` contenant au minimum les colonnes :
  - dataset_size
  - n_variables
  - train_time_seconds
  - n_iterations
  - accuracy
  - precision
  - recall
  - f1_score
  - roc_auc

Exécution (PowerShell) :
```
& .\cicids_env\Scripts\Activate.ps1
python .\scripts\run_experiments_fr.py
```

## D. Visualisations
Un script `scripts/plot_results_fr.py` a été ajouté pour générer les figures demandées à partir de `results/experiments_results.csv` :
- Temps d'entraînement en fonction de la taille du dataset ;
- Nombre d'itérations en fonction de la taille du dataset ;
- Accuracy en fonction de la taille du dataset ;
- F1-Score en fonction de la taille du dataset.

Les figures sont sauvegardées dans `results/` (PNG). Exécution :
```
& .\cicids_env\Scripts\Activate.ps1
python .\scripts\plot_results_fr.py
```

## E. Rapport technique
Rédiger un rapport synthétique (3 à 5 pages) et le sauvegarder dans `results/rapport_technique.md` ou `results/rapport_technique.pdf`.
Contenu recommandé :
- Étapes de préparation des données (résumé des transformations effectuées) ;
- Choix techniques (imputation, downcasting, seuils) ;
- Environnement matériel et logiciel utilisé (CPU/RAM, versions Python & packages) ;
- Résultats obtenus (tableau + figures) ;
- Difficultés rencontrées et pistes d'amélioration.

Le dépôt contient désormais les scripts automatisés pour :
- générer les subsets (`generate_subsets_fr.py`),
- exécuter les expériences et collecter les métriques (`run_experiments_fr.py`),
- tracer les figures (`plot_results_fr.py`).

Si vous souhaitez que j'exécute ces scripts maintenant (générer subsets, lancer les expériences et produire figures + fichier CSV résultats + rapport minimal), répondez "Oui, exécuter" et j'exécuterai les étapes disponibles localement ici.

---

Fichier README mis à jour le: 2026-06-03
