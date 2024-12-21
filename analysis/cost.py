from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from constants import TRAIN_FLOPS, FAMILY, N, AVERAGE_BENCHMARK
from data import merge_model_information, load_experiment

PROJECT_ROOT = Path(__file__).parent.parent

def token_info():   
    df = pd.read_csv(PROJECT_ROOT / "results" / "toks_generated.csv")

    # split the model name into model and checkpoint
    df[['model', 'checkpoint']] = (
        df['model'].str.split("\_", n=1).apply(
            lambda x: pd.Series(x if len(x) > 1 else [x[0], 'main'])
            ))

    agg_df = df.groupby(['model', 'checkpoint']).agg({'reg_tokens': 'mean', 'cot_tokens': 'mean', 'token_ratio': 'mean'})
    agg_df = agg_df.reset_index()

    agg_df = merge_model_information(agg_df)

    agg_df['Standard'] = agg_df['reg_tokens']
    agg_df['Chain of Thought'] = agg_df['cot_tokens']

    return agg_df




if __name__ == "__main__":
    pass
    # tokens_vs_performance()
    # inference_flops_vs_performance()
    train_vs_test_cost()
    # agg_df = token_info()
    # df = agg_df.melt(
    #     id_vars=[TRAIN_FLOPS, FAMILY], value_vars=['Chain of Thought', 'Standard'], var_name='Prompting Type', value_name='Tokens Generated')

    # sns.scatterplot(
    #     x=TRAIN_FLOPS, y='Tokens Generated', data=df, hue='Prompting Type',
    #     style=FAMILY
    #     )

    # plt.show()

    # agg_df['CoT Token Ratio'] = agg_df['Chain of Thought'] / agg_df['Standard']

    # fig = plt.figure(figsize=(15, 8))
    # ax = sns.scatterplot(
    #     x=TRAIN_FLOPS, y='CoT Token Ratio', data=agg_df, hue=D,
    #     style=FAMILY
    #     )

    # ax.set_xscale('log')
    # # add extra space on the right
    # plt.tight_layout(rect=[0, 0.7, 0, 1])  # Reserves 15% of figure for legend and 5% on top/bottom
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # plt.show()