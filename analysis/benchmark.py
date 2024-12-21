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

from constants import (
    D, N, FAMILY, AVERAGE_BENCHMARK, TRAIN_FLOPS,
    BENCHMARKS, ARC, GSM8K, HELLASWAG, MMLU, WINOGRANDE, XWINOGRAD, TRUTHFULQA
)


def merge_model_information(df):
    models = pd.read_csv(PROJECT_ROOT / "models.csv")
    df = df.merge(models, on=['model', 'checkpoint'])

    # add in new columns with human readable names
    df[D] = df['D']
    df[N] = df['N']
    df[FAMILY] = df['family']
    df[AVERAGE_BENCHMARK] = df['Overall']
    df[TRAIN_FLOPS] = df['D'] * df['N'] * 6

    return df

def extract_benchmarks(df, experiment: int, method: str):
    """
    Extracts benchmark-level results from the dataframe.
    """
    if method == "standard" or experiment == 2:
        df = df.rename(columns={
            'arc_challenge_acc_none': ARC,
            'gsm8k_exact_match_flexible-extract': GSM8K,
            'hellaswag_acc_none': HELLASWAG,
            'mmlu_acc_none': MMLU,
            'winogrande_acc_none': WINOGRANDE,
            'xwinograd_acc_none': XWINOGRAD,
            'truthfulqa_mc1_acc_none': TRUTHFULQA
        })
    elif method == "cot" and experiment == 1:
        # calculate the winograd and MMLU overall results
        is_mmlu_col = lambda c: c.startswith('mmlu_flan_cot') and 'flexible-extract' in c
        mmlu_cols = list(filter(is_mmlu_col, df.columns))
        df[MMLU] = df[mmlu_cols].mean(axis=1)

        is_xwinograd_col = lambda c: c.startswith('xwinograd_cot')
        xwinograd_cols = list(filter(is_xwinograd_col, df.columns))
        df[XWINOGRAD] = df[xwinograd_cols].mean(axis=1)

        df = df.rename(columns={
            'arc_challenge_cot_exact_match_flexible-extract': ARC,
            'gsm8k_cot_zeroshot_exact_match_flexible-extract': GSM8K,
            'hellaswag_cot_exact_match_flexible-extract': HELLASWAG,
            'winogrande_cot_exact_match_flexible-extract': WINOGRANDE,
            'truthfulqa_cot_exact_match_flexible-extract': TRUTHFULQA
        })

    df['Overall'] = df[BENCHMARKS].mean(axis=1)

    return df

def load_df(experiment: int, method: str):
    if method not in ["standard", "cot"]:
        raise ValueError(f"Method {method} not supported")

    df = pd.read_csv(PROJECT_ROOT / f"results/csv/experiment_{experiment}/{method}.csv")

    df = extract_benchmarks(df, experiment, method)

    df = df.loc[:, ['model', 'checkpoint', "Overall"] + BENCHMARKS]

    return df

def load_experiment(experiment: int):
    df = load_df(experiment, "standard")
    cot_df = load_df(experiment, "cot")

    df = df.merge(cot_df, on=['model', 'checkpoint'], suffixes=('', '_COT'))
    df = merge_model_information(df)
    return df

def base_bench_by_N(df, outpath, cot=False):
    """
    Baseline Model Benchmark Performance by N.
    """
    bench_col = 'Overall' if not cot else 'Overall_COT'
    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=N, y=bench_col, data=df, style=FAMILY, hue=D, palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Baseline Model Benchmark Performance'
    if cot:
        title += ' (Chain of Thought)'
    ax.set_title(title)

    plt.savefig(outpath)


def base_bench_by_train_flops(df, outpath, cot=False):
    """
    Baseline Benchmark Performance, with train FLOPs on the x-axis.
    """
    bench_col = 'Overall' if not cot else f'Overall_COT'
    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y=bench_col, data=df, style=FAMILY, hue=D, palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Baseline Model Performance by Training FLOPs'
    if cot:
        title += ' (Chain of Thought)'
    ax.set_title(title)

    plt.savefig(outpath)

def cot_improvement_by_train_flops(df, outpath):
    """
    Chain of Thought Improvement by Training FLOPs.
    """
    # pivot df longer so it can be plotted with sns.scatterplot
    df['CoT Improvement'] = df['Overall_COT'] - df['Overall']

    fig = plt.figure(figsize=(15, 8))

    sns.lineplot(
        x=TRAIN_FLOPS, y='CoT Improvement', data=df, 
        color='black', style=FAMILY, alpha=0.3, legend=False
    )
    
    # Original scatterplot on top
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y='CoT Improvement', data=df, style=FAMILY,
        hue=D, palette='flare', hue_norm=LogNorm()
    )
    
    # add a line at y=0
    ax.axhline(0, color='black', linestyle='-', alpha=0.2)

    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Chain of Thought Improvement by Training FLOPs'
    ax.set_title(title)

    plt.savefig(outpath)

def bench_by_train_flops(df, outpath):
    """
    Benchmark Performance with and without Chain of Thought, with train FLOPs on the x-axis.
    """
    # pivot df longer so it can be plotted with sns.scatterplot
    df['Standard'] = df['Overall']
    df['Chain of Thought'] = df['Overall_COT']

    df = df.melt(
        id_vars=['model', FAMILY, TRAIN_FLOPS],
        value_vars=['Standard', 'Chain of Thought'],
        var_name='Method',
        value_name='Benchmark Score'
    )
    print(df.columns)

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y='Benchmark Score', data=df, style=FAMILY,
        hue='Method'
        )


    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Benchmark Performance by Training FLOPs'
    ax.set_title(title)

    plt.savefig(outpath)
def dataset_pc_loadings(outpath):
    """
    Dataset PC Loadings.
    """

    loadings = pd.read_csv(PROJECT_ROOT / "benchmark_loadings.csv", index_col=0)

    fig = plt.figure(figsize=(15, 8))
    ax = sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm')

    ax.set_title('Dataset PC Loadings')

    # add extra space on the left
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(outpath)

def chain_of_thought_gsm8k(df, cot_df, outpath):
    """
    Chain of Thought Performance on GSM8K.
    """
    cot_df['gsm8k_cot'] = cot_df['gsm8k_cot_zeroshot_exact_match_flexible-extract']
    df['GSM8K Score'] = df['gsm8k_exact_match_flexible-extract']

    overlap = df.merge(cot_df, on=['model', 'checkpoint'])
    print(overlap.columns)
    # overlap = overlap.loc[:, ['model', 'checkpoint', 'gsm8k_cot', 'gsm8k_base']]
    overlap[TRAIN_FLOPS] = overlap[f'{TRAIN_FLOPS}_y']
    overlap[FAMILY] = overlap[f'{FAMILY}_y']
    overlap[D] = overlap[f'{D}_y']

    sns.scatterplot(x=TRAIN_FLOPS, y='GSM8K Score', data=overlap)
    sns.scatterplot(x=TRAIN_FLOPS, y='gsm8k_cot', data=overlap)

    overlap['CoT Improvement'] = overlap['gsm8k_cot'] - overlap['GSM8K Score']

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(x=TRAIN_FLOPS, y='CoT Improvement', data=overlap, style=FAMILY, hue=D, palette='flare', hue_norm=LogNorm())
    ax.set_xscale('log')
    ax.set_title('Chain of Thought Improvement on GSM8K')
    # add extra space on the right
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(outpath)

    plt.show()

if __name__ == "__main__":
    sns.set_context("talk")

    EXPERIMENT = 2

    df = load_experiment(EXPERIMENT)
    print(df.head())
    print(df.columns)

    figdir = PROJECT_ROOT / "figures" / f"experiment_{EXPERIMENT}"
    figdir.mkdir(parents=True, exist_ok=True)

    # base_bench_by_N(df, figdir / "figure_1.png")
    # base_bench_by_train_flops(df, figdir / "figure_2.png")

    # base_bench_by_N(df, figdir / "figure_1_cot.png", cot=True)
    # base_bench_by_train_flops(df, figdir / "figure_2_cot.png", cot=True)
    bench_by_train_flops(df, figdir / "standard_vs_cot.png")

    cot_improvement_by_train_flops(df, figdir / "cot_improvement_by_train_flops.png")

    # dataset_pc_loadings(figdir / "dataset_pc_loadings.png")

    # chain_of_thought(figdir / "chain_of_thought.png")
    # base_bench_by_N(df, figdir / "cot_figure_1.png")
    # base_bench_by_train_flops(df, figdir / "cot_figure_2.png")

    # chain_of_thought_gsm8k(df, figdir / "cot_gsm8k.png")
