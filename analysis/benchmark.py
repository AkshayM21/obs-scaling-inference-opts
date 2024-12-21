"""
Displays the performance of the model on the benchmark tasks.
"""
from pathlib import Path

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from constants import (
    D, N, FAMILY, TRAIN_FLOPS, AVERAGE_BENCHMARK
)
from data import load_experiment

PROJECT_ROOT = Path(__file__).parent.parent

TIGHT_LAYOUT_RECT = [0, 0, 0.7, 0.95]

def base_bench_by_N(df, outpath, cot=False):
    """
    Baseline Model Benchmark Performance by N.
    """
    bench_col = 'Overall' if not cot else 'Overall_COT'
    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=N, y=bench_col, data=df, style=FAMILY, hue=D,
        palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
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
        x=TRAIN_FLOPS, y=bench_col, data=df, style=FAMILY, hue=D,
        palette='flare', hue_norm=LogNorm(), sizes=[5],
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
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
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Chain of Thought Improvement by Training FLOPs'
    ax.set_title(title)

    plt.savefig(outpath)

def bench_by_train_flops(df, outpath):
    """
    Benchmark Performance with and without Chain of Thought,
    with train FLOPs on the x-axis.
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

    df[AVERAGE_BENCHMARK] = df['Benchmark Score']

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y=AVERAGE_BENCHMARK, data=df, style=FAMILY,
        hue='Method'
        )


    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Performance with and without Chain of Thought'
    ax.set_title(title)

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
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(outpath)

    plt.show()


def bench_cross_experiment(outpath: Path):
    """
    Compare the average benchmark scores between the two experiments.
    """
    df_exp_1 = load_experiment(1)
    df_exp_2 = load_experiment(2)

    df_exp_1['Evaluation Mode'] = 'Log-Likelihood'
    df_exp_2['Evaluation Mode'] = 'Free-Form Generation'

    df = pd.concat([df_exp_1, df_exp_2])

    fig = plt.figure(figsize=(15, 8))
    ax = sns.scatterplot(
        x=TRAIN_FLOPS, y=AVERAGE_BENCHMARK, data=df,
        style=FAMILY, hue='Evaluation Mode', palette="Set2"
        )
    ax.set_xscale('log')
    # add extra space on the right
    plt.tight_layout(rect=TIGHT_LAYOUT_RECT)  # Reserves 15% of figure for legend and 5% on top/bottom
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # add title
    title = 'Benchmark Scores by Training FLOPs across Experiments'
    ax.set_title(title)
    plt.savefig(outpath)

def bench_performance(experiment: int):
    df = load_experiment(experiment, standard_only=True)

    figdir = PROJECT_ROOT / "figures" / f"experiment_{experiment}"
    figdir.mkdir(parents=True, exist_ok=True)

    base_bench_by_N(df, figdir / "figure_1.png")
    base_bench_by_train_flops(df, figdir / "figure_2.png")

    df = load_experiment(experiment)

    base_bench_by_N(df, figdir / "figure_1_cot.png", cot=True)
    base_bench_by_train_flops(df, figdir / "figure_2_cot.png", cot=True)

    bench_by_train_flops(df, figdir / "standard_vs_cot.png")

    cot_improvement_by_train_flops(df, figdir / "cot_improvement_by_train_flops.png")

if __name__ == "__main__":
    sns.set_context("poster")

    figdir = PROJECT_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    bench_cross_experiment(figdir / "bench_cross_experiment.png")

    bench_performance(experiment=1)
    bench_performance(experiment=2)

    # dataset_pc_loadings(figdir / "dataset_pc_loadings.png")

    # chain_of_thought(figdir / "chain_of_thought.png")
    # base_bench_by_N(df, figdir / "cot_figure_1.png")
    # base_bench_by_train_flops(df, figdir / "cot_figure_2.png")

    # chain_of_thought_gsm8k(df, figdir / "cot_gsm8k.png")
