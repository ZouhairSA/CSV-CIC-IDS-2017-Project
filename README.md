# CSV-CIC-IDS-2017-Project

Rapport technique et pipeline reproductible pour l'expérimentation (classification binaire) sur le dataset CIC-IDS-2017.

## Contenu
- `scripts/` : scripts Python pour préparation, génération de sous-ensembles, entraînement et visualisation.
- `results/` : artefacts générés (figures, subsets, rapport). **Les gros fichiers (datasets bruts, environnements virtuels) sont exclus du dépôt.**
- `README.md`, `README.fr.md` : documentation du projet.

## Objectifs
- Préparer et nettoyer le dataset CIC-IDS-2017.
- Générer des sous-ensembles de tailles variées (1k → 500k).
- Exécuter des expérimentations reproductibles (Logistic Regression) et produire métriques et figures.
- Fournir un rapport technique (LaTeX / PDF) avec toutes les métriques et figures.

## Prérequis
- Windows, PowerShell
- Python 3.10+ (venv fourni `cicids_env/` — ne pas committer)
- Recommandé : MiKTeX / TeXLive pour compiler le `.tex` en PDF

## Installation rapide
1. Activer l'environnement virtuel local (si vous utilisez le venv fourni) :

```powershell
Set-Location "C:\Users\sabyo\Desktop\CIC-IDS- 2017-Project"
.\cicids_env\Scripts\Activate.ps1
pip install -r requirements.txt   # si vous préférez réinstaller les dépendances
```

2. Préparer les données (placer les CSV originaux dans `data/raw/`) :

```powershell
python .\scripts\prepare_cicids2017_fr.py
```

3. Générer les sous-ensembles :

```powershell
python .\scripts\generate_subsets_fr.py --sizes 1000,5000,10000,20000,50000,100000,200000,500000
```

4. Lancer les expérimentations et générer les figures :

```powershell
python .\scripts\run_experiments_fr.py
python .\scripts\plot_results_fr.py
```

5. Compiler le rapport LaTeX (optionnel) :

```powershell
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -halt-on-error" results\rapport_technique.tex
```

## Ajouter des images de l'entraînement (figures)
- Les figures générées automatiquement par `plot_results_fr.py` sont écrites dans `results/` :
  - `results/plot_train_time_seconds.png`
  - `results/plot_n_iterations.png`
  - `results/plot_accuracy.png`
  - `results/plot_f1_score.png`

- Aperçu (les images sont incluses depuis `results/` et s'affichent directement sur GitHub) :

  ![Temps d'entraînement](results/plot_train_time_seconds.png)

  ![Nombre d'itérations](results/plot_n_iterations.png)

  ![Accuracy](results/plot_accuracy.png)

  ![F1-Score](results/plot_f1_score.png)

- Pour inclure vos propres captures d'écran ou photos d'entraînement : placez-les dans `results/figures/` puis ajoutez-les au dépôt.
  - IMPORTANT : si les images sont volumineuses, utilisez Git LFS plutôt que de committer directement. Installer et initialiser Git LFS :

```powershell
git lfs install
git lfs track "results/figures/*"
git add .gitattributes
```

Puis `git add results/figures/<votre_image>.png` et commit/push.

## Bonnes pratiques Git
- Ne commitez pas `data/raw/`, `cicids_env/`, `archive.zip` ni `results/` contenant datasets bruts. Ces dossiers figurent déjà dans `.gitignore`.
- Si vous avez committé de gros fichiers par erreur, utilisez `bfg` ou `git filter-repo` pour purifier l'historique avant push.

## Licence
- Documentez ici la licence souhaitée (ex. MIT). Si aucune licence n'est fournie, le projet reste © auteur.

---

Si vous voulez que je remplisse aussi `README.fr.md` en français complet et que j'ajoute automatiquement les petites figures présentes localement (copie dans le dépôt et config Git LFS si nécessaire), répondez `README-fr+figs, oui` et j'exécuterai les étapes. Otherwise, je peux only create README.md (done).
