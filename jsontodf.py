import pandas as pd
import json
import glob
import os
from typing import Dict, Any

def extract_model_info(filename: str) -> Dict[str, str]:
    """Extract model name and checkpoint information from filename."""
    base = os.path.basename(filename)
    base = base.replace('.json', '')
    
    # Split on potential checkpoint marker
    parts = base.split('_')
    model_name = parts[0]
    checkpoint = parts[1] if len(parts) > 1 else 'main'
    
    return {
        'model': model_name,
        'checkpoint': checkpoint
    }

def flatten_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Flatten nested metrics structure into key-value pairs."""
    flattened = {}
    
    # Process main results
    for task, metrics in data['results'].items():
        for metric_name, value in metrics.items():
            # Skip alias and stderr metrics
            if metric_name == 'alias' or 'stderr' in metric_name:
                continue
            # Create column name from task and metric
            column_name = f"{task}_{metric_name}".replace(',', '_')
            flattened[column_name] = value
            
    return flattened

def process_results_files(directory: str) -> pd.DataFrame:
    """Process all JSON files in directory and return a DataFrame."""
    all_results = []
    
    # Get all JSON files in directory
    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    for file_path in json_files:
        try:
            # Extract model info from filename
            model_info = extract_model_info(file_path)
            
            # Read and parse JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Flatten metrics
            metrics = flatten_metrics(data)
            
            # Combine model info and metrics
            result = {**model_info, **metrics}
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Set model and checkpoint as index
    df = df.set_index(['model', 'checkpoint'])
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    results_dir = "cot_results/"
    df = process_results_files(results_dir)
    df.to_csv("cot_results.csv")
    print(df.head())