import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm

from constants import TRAIN_FLOPS, AVERAGE_BENCHMARK, FAMILY, N
from cost import token_info
from data import load_experiment
from train_equivalent import load_train_equivalent_flops
from scipy.optimize import curve_fit


#fits benchmark performances vs inference tokens with isoflop curves (figure 7)
def isoflop_tokens_vs_performance():
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

    def isoflop_model(X, a, b, c):
        tokens, flops = X
        return a * np.log(flops) + b * tokens + c

    # Adjust figure size to be wider than tall
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Scatter plot
    scatter = sns.scatterplot(
        x='Tokens Generated', y=AVERAGE_BENCHMARK, data=df, 
        hue=TRAIN_FLOPS, palette='flare', hue_norm=LogNorm(), 
        style='Prompting Type'
    )
    

    # Add regression lines
    legend_elements = []
    legend_elements.extend(scatter.get_legend_handles_labels()[0])
    for opt, color in zip(['Standard', 'Chain of Thought'], ['blue', 'green']):   
        df_opt = df[df['Prompting Type'] == opt]
        reg = sns.regplot(
            x='Tokens Generated', y=AVERAGE_BENCHMARK, data=df_opt, 
            scatter=False, ci=None,
            line_kws={'linestyle': '-', 'color': color, 'alpha': 0.8}
        )
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-',
                                        label=f'{opt} (best fit)'))
        
    # Add fitted isoflop curves
    flop_values = [1e20, 1e21, 1e22]
    token_range = np.linspace(5, 150, 146)
    

    standard_colors = ['#ff8888', '#ff4444', '#ff0000']
    cot_colors = ['#888888', '#444444', '#000000']

    for i, prompt_type in enumerate(['Standard', 'Chain of Thought']):
        df_prompt = df[df['Prompting Type'] == prompt_type]
        X_prompt = np.vstack((
            df_prompt['Tokens Generated'].values,
            df_prompt[TRAIN_FLOPS].values
        ))
        y_prompt = df_prompt[AVERAGE_BENCHMARK].values
        
        popt_prompt, _ = curve_fit(isoflop_model, X_prompt, y_prompt, p0=[0.01, 0.01, 0.3])
        print(prompt_type)
        print(popt_prompt)
        
        colors = standard_colors if prompt_type == 'Standard' else cot_colors
        for j, (flops, color) in enumerate(zip(flop_values, colors)):
            X_pred = np.vstack((
                token_range,
                np.full_like(token_range, flops)
            ))
            scores_pred = isoflop_model(X_pred, *popt_prompt)
            line = plt.plot(token_range, scores_pred, '--', color=color, alpha=0.5)[0]

            standard_index = np.abs(scores_pred-0.33).argmin()
            
            # Adjust text label position
            if prompt_type == 'Standard':
                plt.text(X_pred[0][standard_index], 0.33, f'{flops:.0e}', 
                    color=color, fontsize=8, alpha=0.7)
            else:
                plt.text(120, scores_pred[-1], f'{flops:.0e}', 
                    color=color, fontsize=8, alpha=0.7)
            
            if j == 1:  # Middle curve for legend
                legend_elements.append(plt.Line2D([0], [0], color=colors[1], 
                    linestyle='--', label=f'{prompt_type} IsoFLOP'))

    plt.xscale('log')
    plt.xlabel('Inference Tokens Generated (Log Scale)')
    plt.ylabel('Average Benchmark Score')
    
    # Tighter axis limits
    plt.xlim(5, 150)
    plt.ylim(0.28, 0.35)
    
    # Format x-axis ticks to prevent overlap
    # ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    
    # Move legend outside to the right
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), frameon=True)
    
    # Adjust layout with more space for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.7, top=0.95)

    plt.show()


if __name__=="__main__":
    isoflop_tokens_vs_performance()