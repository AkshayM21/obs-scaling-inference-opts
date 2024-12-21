"""
Run PCA on the benchmark results.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from constants import (
    BENCHMARKS, MMLU, ARC, HELLASWAG, TRUTHFULQA, WINOGRANDE, GSM8K
    )
from data import load_experiment


sns.set_context("talk")

PROJECT_ROOT = Path(__file__).parent.parent
N_COMPONENTS = 3


def run_pca(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Run PCA on the benchmark results.
    """
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(array)
    # get the factor loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return pca.transform(array), loadings, pca

def our_pca():
    """
    Compute PCA on the base benchmark results from our experiments.
    """
    df = load_experiment(1)
    print(df.head())
    # df = pd.read_csv(PROJECT_ROOT / "benchmark_results.csv")

    # bench_scores = df[BENCHMARKS].to_numpy()
    # pcs, loadings, pca = run_pca(bench_scores)

    # # store the loadings to a csv file along with the benchmarks
    # # benchmark name, loading for pc_1, loading for pc_2, loading for pc_3
    # loadings_df = pd.DataFrame(loadings, columns=[f'pc_{i+1}' for i in range(N_COMPONENTS)])
    # loadings_df.index = BENCHMARKS
    # loadings_df.to_csv(PROJECT_ROOT / "benchmark_loadings.csv")

    # pc_cols = [f'pc_{i+1}' for i in range(N_COMPONENTS)]
    # for i, pc_col in enumerate(pc_cols):
    #     df[pc_col] = pcs[:, i]

    # df = df[['model', 'checkpoint', *pc_cols]]

    # df.to_csv(PROJECT_ROOT / "benchmark_pcs.csv", index=False)

def ruan_pca():
    """
    Recompute PCA on the results from Ruan et al. (2024),
    without the HumanEval benchmark.
    """
    df = pd.read_csv(PROJECT_ROOT / "results/csv/ruan.csv")

    df.rename(columns={
        'MMLU': MMLU,
        'ARC-C': ARC,
        'HellaSwag': HELLASWAG,
        'TruthfulQA': TRUTHFULQA,
        'Winograd': WINOGRANDE, # typo in the paper
        'GSM8K': GSM8K
    }, inplace=True)

    # throw out any rows with NaN values in the BENCHMARKS columns
    df = df.dropna(subset=BENCHMARKS)

    arr = df[BENCHMARKS].to_numpy()

    _, loadings, pca = run_pca(arr)

    # save loadings to a csv file
    loadings_df = pd.DataFrame(loadings, columns=[f'pc_{i+1}' for i in range(N_COMPONENTS)])
    loadings_df.index = BENCHMARKS
    loadings_df.to_csv(PROJECT_ROOT / "results" / "pca" / "ruan_loadings.csv")

def dataset_pc_loadings(outpath):
    """
    Dataset PC Loadings.
    """

    loadings = pd.read_csv(PROJECT_ROOT / "results" / "pca" / "ruan_loadings.csv", index_col=0)

    fig = plt.figure(figsize=(15, 8))
    ax = sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.92, vmax=0.83)

    ax.set_title('Dataset PCA Loadings, Reproduced from Ruan et al. (2024)')

    # add extra space on the left
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(outpath)

if __name__ == "__main__":
    ruan_pca()
    dataset_pc_loadings(PROJECT_ROOT / "figures" / "ruan_loadings.png")