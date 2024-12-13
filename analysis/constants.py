from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# create temporary df with renamed columns
D = 'Training Tokens (D)'
N = 'Model Parameters (N)'
FAMILY = 'Model Family'
AVERAGE_BENCHMARK = 'Average Benchmark Score'
TRAIN_FLOPS = 'Training FLOPs ($6ND$)'

BENCHMARKS = [
    'arc_challenge_acc_none',
    'gsm8k_exact_match_flexible-extract',
    'hellaswag_acc_none',
    'mmlu_acc_none',
    'winogrande_acc_none',
    'xwinograd_acc_none',
    'truthfulqa_mc1_acc_none'
]

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