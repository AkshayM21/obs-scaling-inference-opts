import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm

from constants import TRAIN_FLOPS, AVERAGE_BENCHMARK, FAMILY, N
from cost import token_info
from data import load_experiment
from train_equivalent import load_train_equivalent_flops

def tokens_vs_performance():
    performance_df = load_experiment(2)
    tokens_df = token_info()

    performance_df = performance_df.merge(
        tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY])

    df = performance_df.melt(
        id_vars=[TRAIN_FLOPS, FAMILY, AVERAGE_BENCHMARK],
        value_vars=['Chain of Thought', 'Standard'],
        var_name='Prompting Type',
        value_name='Tokens Generated'
        )


    sns.set_context("poster")
    fig = plt.figure(figsize=(15, 8))

    ax = sns.scatterplot(
        x='Tokens Generated', y=AVERAGE_BENCHMARK, data=df, hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm(), style='Prompting Type'
        )

    for opt, linestyle in zip(df['Prompting Type'].unique(), ['-', '--']):   
        df_opt = df[df['Prompting Type'] == opt]
        sns.regplot(
            x='Tokens Generated', y=AVERAGE_BENCHMARK, data=df_opt, scatter=False, ci=None,
            line_kws={'linestyle': linestyle, 'color': 'black', 'alpha': 0.5}
            )

    # ensure legend is on the right
    # with plenty of space on the right
    plt.tight_layout(rect=[0, 0, 0.67, 1])
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # add a best fit line per family
    # for family in df[FAMILY].unique():
    #     df_family = df[df[FAMILY] == family]
    #     sns.regplot(x='Tokens Generated', y=AVERAGE_BENCHMARK, data=df_family, scatter=True, color='black')

    plt.show()


def inference_flops_vs_performance():
    performance_df = load_experiment(2)
    tokens_df = token_info()

    performance_df = performance_df.merge(
        tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    print(performance_df.columns)

    df = performance_df.melt(
        id_vars=[TRAIN_FLOPS, FAMILY, N, 'Overall', 'Overall_COT'],
        value_vars=['Chain of Thought', 'Standard'],
        var_name='Prompting Type',
        value_name='Tokens Generated'
        )

    # select 'Overall' or 'Overall_COT' depending on the prompting type
    df[AVERAGE_BENCHMARK] = df.apply(lambda row: row['Overall'] if row['Prompting Type'] == 'Standard' else row['Overall_COT'], axis=1)

    # 2N FLOPs per forward pass
    df['Average Inference FLOPs per Request'] = 2 * df[N] * df['Tokens Generated']

    sns.scatterplot(
        x='Average Inference FLOPs per Request', y=AVERAGE_BENCHMARK, data=df, hue='Prompting Type',
        style=FAMILY
        )

    # use log scale for x-axis
    plt.xscale('log')

    plt.show()

def cost_vs_performance():
    performance_df = load_experiment(2)
    tokens_df = token_info()

    performance_df = performance_df.merge(
        tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    # cost effectiveness = delta performance / delta cost

def train_vs_test_cost_ratio():
    train_equivalent_flops = load_train_equivalent_flops()

    tokens_df = token_info()

    df = train_equivalent_flops.merge(
        tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    # x_axis = 'Increase in Inference FLOPs with Chain of Thought'
    # df[x_axis] = (df['Chain of Thought'] - df['Standard']) * 2 * df[N]

    df['train_equiv_ratio'] = df['Train Equivalent FLOPs'] / df[TRAIN_FLOPS]

    sns.regplot(
        x='token_ratio', y='train_equiv_ratio', data=df,scatter=False, ci=None
        )
    sns.scatterplot(
        x='token_ratio', y='train_equiv_ratio', data=df, style=FAMILY, hue=TRAIN_FLOPS,palette='flare', hue_norm=LogNorm()
        )

    # dashed line at y = 1
    plt.axhline(y=1, color='black', linestyle='--')

    plt.show()

def train_vs_test_cost_absolute():
    train_equivalent_flops = load_train_equivalent_flops()

    tokens_df = token_info()

    df = train_equivalent_flops.merge(
        tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    # x_axis = 'Increase in Inference FLOPs with Chain of Thought'
    # df[x_axis] = (df['Chain of Thought'] - df['Standard']) * 2 * df[N]

    df['Delta Log Train FLOPs to Achieve Same Performance'] = np.log10(df['Train Equivalent FLOPs']) - np.log10(df[TRAIN_FLOPS])
    print(df['Delta Log Train FLOPs to Achieve Same Performance'])

    df['Delta Inference FLOPs per Request'] = df['Chain of Thought'] - df['Standard']
    print(df['Delta Inference FLOPs per Request'])

    sns.scatterplot(
        x='Delta Inference FLOPs per Request', y='Delta Log Train FLOPs to Achieve Same Performance', data=df, style=FAMILY, hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm()
        )


    # # dashed line at y = 1
    # plt.axhline(y=1, color='black', linestyle='--')

    plt.show()

def akshay_plot(cot=False):
    # x-axis: log (train flops) * inference_tokens
    # y-axis: average benchmark score
    # style: family

    df = load_experiment(2)
    tokens_df = token_info()

    df = df.merge(tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    if cot:
        df['log(Train FLOPs) * Average Inference Tokens'] = np.log10(df[TRAIN_FLOPS]) * df['Chain of Thought']
    else:
        df['log(Train FLOPs) * Average Inference Tokens'] = np.log10(df[TRAIN_FLOPS]) * df['Standard']

    outcome = 'Overall' if not cot else 'Overall_COT'
    sns.scatterplot(x='log(Train FLOPs) * Average Inference Tokens', y=outcome, data=df, style=FAMILY, hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm())
    sns.lineplot(x='log(Train FLOPs) * Average Inference Tokens', y=outcome, data=df, style=FAMILY, hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm())

    plt.show()

# def akshay_plot_diff():
#     df = load_experiment(2)
#     tokens_df = token_info()

#     df = df.merge(tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

#     df['log(Train FLOPs) * Average Increase in Inference Tokens'] = np.log10(df[TRAIN_FLOPS]) * df['Chain of Thought']
#     df['log(Train FLOPs) * Average Inference Tokens'] = np.log10(df[TRAIN_FLOPS]) * df['Standard']

#     sns.scatterplot(x='log(Train FLOPs) * Average Inference Tokens', y='Overall', data=df, style=FAMILY, hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm())

#     plt.show()

def intersecting_lines():
    # inference requests on the x axis
    # total cost on the y axis

    df = load_train_equivalent_flops()
    tokens_df = token_info()

    df = df.merge(tokens_df, on=['model', 'checkpoint', TRAIN_FLOPS, FAMILY, N])

    row = df.loc[(df['model'] == 'OLMo-1B-0724-hf') & (df['checkpoint'] == 'main')].iloc[0]
    print(row)


    req_counts = np.logspace(0, 15, 100)

    cot_cost = row[TRAIN_FLOPS] + (2 * req_counts * row['Chain of Thought'] * row[N])

    standard_cost = row['Train Equivalent FLOPs'] + (2 * req_counts * row['Standard'] * row[N])

    # analytically find the point where the two lines intersect


    plt.plot(req_counts, cot_cost, label='Chain of Thought')
    plt.plot(req_counts, standard_cost, label='Standard')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Number of Inference Requests')
    plt.ylabel('Total Cost in FLOPs (Train + Inference)')

    # add in a dashed vertical line at 

    plt.legend()

    plt.show()


if __name__ == "__main__":
    # tokens_vs_performance()
    intersecting_lines()
    # inference_flops_vs_performance()
    # train_vs_test_cost_ratio()
    # train_vs_test_cost_absolute()
    # akshay_plot(cot=False)
    # akshay_plot(cot=True)