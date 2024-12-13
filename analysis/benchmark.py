"""
Displays the performance of the model on the benchmark tasks.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).parent.parent

from constants import BENCHMARKS, D, N, FAMILY, AVERAGE_BENCHMARK, TRAIN_FLOPS


def setup_df():
    df = pd.read_csv(PROJECT_ROOT / "benchmark_results.csv")


    df = df.loc[:, ['model', 'checkpoint'] + BENCHMARKS]

    # row-wise mean of the top-level benchmarks
    df['overall'] = df[BENCHMARKS].mean(axis=1)

    models = pd.read_csv(PROJECT_ROOT / "models.csv")
    df = df.merge(models, on=['model', 'checkpoint'])

    # add in new columns with human readable names
    df[D] = df['D']
    df[N] = df['N']
    df[FAMILY] = df['family']
    df[AVERAGE_BENCHMARK] = df['overall']
    df[TRAIN_FLOPS] = df['D'] * df['N'] * 6

    return df


def figure_1(df, outpath):
    """
    Figure 1: Baseline Model Benchmark Performance.
    """

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=N, y=AVERAGE_BENCHMARK, data=df, style=FAMILY, hue=D, palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    ax.set_title('Baseline Model Benchmark Performance')

    plt.savefig(outpath)


def figure_2(df, outpath):
    """
    Figure 2: Baseline Benchmark Performance, with train FLOPs on the x-axis.
    """

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y=AVERAGE_BENCHMARK, data=df, style=FAMILY, hue=D, palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    ax.set_title('Baseline Model Performance by Training FLOPs')

    plt.savefig(outpath)


if __name__ == "__main__":
    sns.set_context("talk")

    df = setup_df()

    figdir = PROJECT_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    figure_1(df, figdir / "figure_1.png")
    figure_2(df, figdir / "figure_2.png")
