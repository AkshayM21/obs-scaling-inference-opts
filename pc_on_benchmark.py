import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space

def main():
    # Load data
    pcs_df = pd.read_csv('benchmark_pcs.csv')
    cot_df = pd.read_csv('cot_results.csv')

    # Extract the GSM8K benchmark scores
    benchmark_col = 'gsm8k_cot_zeroshot_exact_match_strict-match'

    # Merge datasets on model name
    merged_df = pd.merge(pcs_df, cot_df[['model', benchmark_col]], on='model', how='inner')

    # Extract feature matrix S and benchmark scores B
    S = merged_df[['pc_1', 'pc_2', 'pc_3']].values
    B = merged_df[benchmark_col].values

    def fit_gamma(S, B):
        """
        Fit γ vector to minimize ||B - γᵀS||²
        
        Args:
            S: capability vectors (n_models × K)
            B: benchmark scores (n_models,)
            
        Returns:
            gamma: optimal γ vector (K,)
            mse: mean squared error of the fit
        """
        K = S.shape[1]
        
        # Solve using ordinary least squares: γ = (SᵀS)⁻¹SᵀB
        gamma = np.linalg.lstsq(S, B, rcond=None)[0]
        
        # Calculate predictions and error
        B_pred = S @ gamma
        mse = np.mean((B - B_pred) ** 2)
        r2 = 1 - np.sum((B - B_pred) ** 2) / np.sum((B - np.mean(B)) ** 2)
        
        return gamma, mse, r2

    # Fit the model
    gamma, mse, r2 = fit_gamma(S, B)

    # Create results dataframe
    results_df = pd.DataFrame({
        'Model': merged_df['model'] + ' (' + merged_df['checkpoint'] + ')',
        'Actual_Score': B,
        'Predicted_Score': S @ gamma
    })

    # Print results
    print("Fitted γ vector:")
    print(f"γ₁ = {gamma[0]:.6f}")
    print(f"γ₂ = {gamma[1]:.6f}")
    print(f"γ₃ = {gamma[2]:.6f}")
    print("\nModel performance:")
    print(f"Mean squared error: {mse:.6f}")
    print(f"R² score: {r2:.6f}")

    # Print predictions
    print("\nModel Predictions:")
    print(results_df.to_string(index=False))

    # Compute correlation between actual and predicted scores
    correlation = np.corrcoef(B, S @ gamma)[0, 1]
    print(f"\nCorrelation between actual and predicted scores: {correlation:.6f}")

    # Optional: Plot actual vs predicted scores
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.scatter(B, S @ gamma, alpha=0.6)
        plt.plot([min(B), max(B)], [min(B), max(B)], 'r--', label='Perfect prediction')
        plt.xlabel('Actual GSM8K Score')
        plt.ylabel('Predicted Score')
        plt.title('Actual vs Predicted GSM8K Scores')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('figures/gsm8k_prediction.png', bbox_inches='tight')
    except ImportError:
        print("\nMatplotlib not available for plotting")

if __name__=="__main__":
    main()