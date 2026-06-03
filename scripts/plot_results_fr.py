"""
Génère les visualisations demandées à partir de `results/experiments_results.csv`.
- Sauvegarde PNG dans `results/`.
- Graphes : train_time, n_iterations, accuracy, f1_score vs dataset_size.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def plot_series(df, x, y, out_path, logx=True, xlabel='Taille dataset', ylabel=None, title=None):
    plt.figure(figsize=(6,4))
    plt.plot(df[x], df[y], marker='o')
    if logx:
        plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or y)
    plt.title(title or y)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f'Écrit {out_path}')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(script_dir, '..', 'results'))
    inp = os.path.join(results_dir, 'experiments_results.csv')
    if not os.path.exists(inp):
        logging.error('Aucun fichier experiments_results.csv trouvé dans results/')
        return
    df = pd.read_csv(inp)
    df = df.sort_values('dataset_size')

    plots = [
        ('train_time_seconds', 'Temps entraînement (s)'),
        ('n_iterations', 'Nombre d\'itérations'),
        ('accuracy', 'Accuracy'),
        ('f1_score', 'F1-Score')
    ]
    for col, label in plots:
        out = os.path.join(results_dir, f'plot_{col}.png')
        plot_series(df, 'dataset_size', col, out, logx=True, xlabel='Taille dataset', ylabel=label)

    logging.info('Toutes les figures générées')


if __name__ == '__main__':
    main()
