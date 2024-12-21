from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import TRAIN_FLOPS, AVERAGE_BENCHMARK, FAMILY
from cost import token_info
from data import load_experiment

PROJECT_ROOT = Path(__file__).parent.parent

def train_equivalent():
    df = load_experiment(2)

    df['log_train_flops'] = np.log(df[TRAIN_FLOPS])

    regression_results = []

    # regress average benchmark score on train FLOPs within each family.
    for family in df[FAMILY].unique():
        if family == ' llama':
            continue
        family_df = df[df[FAMILY] == family]
        X = family_df['log_train_flops'].to_numpy().reshape(-1, 1)
        # average benchmark score WITHOUT chain of thought
        y = family_df['Overall'].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        print(f"Family: {family}, R-squared: {model.score(X, y)}")

        b = model.coef_.item()
        a = model.intercept_.item()
        regression_results.append({
            'family': family,
            'b': b,
            'a': a
        })


        # back out the train FLOPs required to match the average
        # CoT benchmark score
        y_cot = family_df['Overall_COT']

        train_equivalent_flops = np.exp((y_cot - a) / b)
        print(f"Train Equivalent FLOPs: {train_equivalent_flops}")

        df.loc[df[FAMILY] == family, 'Train Equivalent FLOPs'] = train_equivalent_flops

    regression_results = pd.DataFrame(regression_results)
    regression_results.to_csv(PROJECT_ROOT / "results/regression.csv", index=False)

    df.to_csv(PROJECT_ROOT / "results/train_equivalent_flops.csv", index=False)


def load_train_equivalent_flops():
    fn = PROJECT_ROOT / "results/train_equivalent_flops.csv"
    try:
        return pd.read_csv(fn)
    except FileNotFoundError:
        train_equivalent()
        return pd.read_csv(fn)

if __name__ == "__main__":
    train_equivalent()