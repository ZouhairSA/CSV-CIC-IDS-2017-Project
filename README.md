# CSV-CIC-IDS-2017-Project

Projet reproductible : préparation, sous-échantillonnage et expérimentation (classification binaire) sur le dataset CIC-IDS-2017.

## Contenu
- `scripts/` : scripts Python (préparation, génération de subsets, entraînement, visualisation).
- `results/` : artefacts générés (figures, subsets, rapport). Certains gros fichiers sont suivis avec Git LFS.
- `data/raw/` : CSV originaux (non inclus dans le dépôt par défaut).

## Objectifs
- Nettoyer et préparer CIC-IDS-2017.
- Générer des sous-ensembles (1k → 500k) pour évaluations rapides et scalabilité.
- Exécuter des expériences reproducibles (Logistic Regression) et produire métriques et figures.
- Rédiger un rapport technique (LaTeX / PDF) contenant résultats et visualisations.

## Prérequis
- Windows (PowerShell recommandé)
- Python 3.10+ (un venv `cicids_env/` est fourni localement — ne pas committer)
- Recommandé : MiKTeX / TeXLive pour compiler le `.tex` en PDF

## Installation rapide
1. Se placer dans le répertoire du projet et activer le venv (optionnel si vous utilisez le venv fourni) :

```powershell
Set-Location "C:\Users\sabyo\Desktop\CIC-IDS- 2017-Project"
.\cicids_env\Scripts\Activate.ps1
pip install -r requirements.txt  # si vous préférez réinstaller les dépendances
```

2. Préparer les données (si vous partez des CSV bruts) :

```powershell
python .\scripts\prepare_cicids2017_fr.py
```

3. Générer les sous-ensembles (exemple) :

```powershell
python .\scripts\generate_subsets_fr.py --sizes 1000,5000,10000,20000,50000,100000,200000,500000
```

4. Lancer les expérimentations et tracer les résultats :

```powershell
python .\scripts\run_experiments_fr.py
python .\scripts\plot_results_fr.py
```

5. (Optionnel) Compiler le rapport LaTeX :

```powershell
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -halt-on-error" results\rapport_technique.tex
```

## Galerie — figures d'entraînement
Les figures générées sont présentes dans `results/` et s'affichent directement sur GitHub :

- `results/plot_train_time_seconds.png` — temps d'entraînement
- `results/plot_n_iterations.png` — nombre d'itérations
- `results/plot_accuracy.png` — accuracy
- `results/plot_f1_score.png` — F1-score

![Temps d'entraînement](results/plot_train_time_seconds.png)

![Nombre d'itérations](results/plot_n_iterations.png)

![Accuracy](results/plot_accuracy.png)

![F1-Score](results/plot_f1_score.png)

> Si une image ne s'affiche pas sur GitHub : vérifiez qu'elle est bien commitée et que son chemin (sensible à la casse) correspond exactement au chemin utilisé ici.

## Notes Git & Git LFS
- Par défaut, `data/raw/`, les environnements virtuels et les gros fichiers sont ignorés par `.gitignore`.
- Certains artefacts volumineux (par ex. `results/subset_*.parquet`) ont été ajoutés avec Git LFS pour permettre le partage. Après clonage du dépôt, récupérer les vrais fichiers LFS :

```powershell
git lfs install
git lfs pull
```

- Si vous ne voulez pas stocker de gros fichiers dans le dépôt, supprimez-les localement et/ou utilisez BFG/git-filter-repo pour purifier l'historique avant push.

## Fichiers importants
- `results/experiments_results.csv` — métriques agrégées (accuracy, F1, temps, itérations, ROC AUC).
- `results/rapport_technique.tex` — source LaTeX du rapport.
- `results/subset_*.parquet` — sous-ensembles (ils peuvent être volumineux).

## Bonnes pratiques
- Ne commitez pas `data/raw/`, `cicids_env/` ni `archive.zip`.
- Pour partager des images ou petites figures, commitez-les dans `results/` (ou `results/figures/`). Pour des fichiers volumineux (>50–100MB), utilisez Git LFS.
- Si vous avez déjà committé de gros fichiers par erreur, utilisez BFG ou `git filter-repo` avant de pousser pour éviter les erreurs de push.

---

Si vous souhaitez que j'améliore aussi `README.fr.md` ou que je supprime les gros `subset_*.parquet` du dépôt (et ne laisse que les PNG), répondez simplement :
- `README-fr, oui` pour créer/mettre à jour `README.fr.md`.
- `retirer subsets, oui` pour retirer les subsets volumineux du dépôt (je préparerai la procédure sûre).
