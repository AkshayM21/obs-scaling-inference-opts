from pathlib import Path

import pandas as pd

from constants import (
    BENCHMARKS, ARC, GSM8K, HELLASWAG, MMLU, WINOGRANDE, XWINOGRAD, TRUTHFULQA,
    D, N, FAMILY, TRAIN_FLOPS, AVERAGE_BENCHMARK
)

PROJECT_ROOT = Path(__file__).parent.parent

def merge_model_information(df):
    models = pd.read_csv(PROJECT_ROOT / "models.csv")
    df = df.merge(models, on=['model', 'checkpoint'])

    # add in new columns with human readable names
    def _add_if_exists(df, col, new_col):
        if col in df.columns:
            df[new_col] = df[col]

    _add_if_exists(df, 'D', D)
    _add_if_exists(df, 'N', N)
    _add_if_exists(df, 'family', FAMILY)
    _add_if_exists(df, 'Overall', AVERAGE_BENCHMARK)

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

def load_experiment(experiment: int, standard_only: bool = False):
    df = load_df(experiment, "standard")
    cot_df = load_df(experiment, "cot")

    if not standard_only:
        df = df.merge(cot_df, on=['model', 'checkpoint'], suffixes=('', '_COT'))

    df = merge_model_information(df)
    return df