"""
Run PCA on the benchmark results.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent

from constants import BENCHMARKS

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

if __name__ == "__main__":
    df = pd.read_csv(PROJECT_ROOT / "benchmark_results.csv")

    bench_scores = df[BENCHMARKS].to_numpy()
    pcs, loadings, pca = run_pca(bench_scores)

    # store the loadings to a csv file along with the benchmarks
    # benchmark name, loading for pc_1, loading for pc_2, loading for pc_3
    loadings_df = pd.DataFrame(loadings, columns=[f'pc_{i+1}' for i in range(N_COMPONENTS)])
    loadings_df.index = BENCHMARKS
    loadings_df.to_csv(PROJECT_ROOT / "benchmark_loadings.csv")

    pc_cols = [f'pc_{i+1}' for i in range(N_COMPONENTS)]
    for i, pc_col in enumerate(pc_cols):
        df[pc_col] = pcs[:, i]

    df = df[['model', 'checkpoint', *pc_cols]]

    df.to_csv(PROJECT_ROOT / "benchmark_pcs.csv", index=False)
