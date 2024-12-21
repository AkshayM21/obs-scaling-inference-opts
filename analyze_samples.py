import json
import os
import pandas as pd
import numpy as np

def analyze_model_outputs():
    # Initialize dictionaries to store results
    results = {
        'cot': {},  # Chain of thought results
        'regular': {}  # Baseline results
    }
    
    # Process both COT and regular samples
    for run_type in ['cot', 'regular']:
        path = os.path.join('samples', run_type)
        
        # Process each JSON file in the directory
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                with open(os.path.join(path, filename), 'r') as f:
                    data = json.load(f)
                
                # Group entries by task name with special handling for mmlu and xwinograd
                task_groups = {}
                for key, entry in data.items():
                    task_name = entry.get('task_name', '')
                    if task_name:
                        # Combine all mmlu subtasks
                        if task_name.startswith('mmlu_'):
                            task_name = 'mmlu'
                        # Combine all xwinograd subtasks
                        elif task_name.startswith('xwinograd_'):
                            task_name = 'xwinograd'
                            
                        if task_name not in task_groups:
                            task_groups[task_name] = []
                        task_groups[task_name].append(entry)
                
                # Calculate metrics for each task
                model_name = filename.replace('.json', '')
                if model_name not in results[run_type]:
                    results[run_type][model_name] = {}
                
                for task_name, entries in task_groups.items():
                    # Calculate word and token counts
                    word_counts = []
                    token_counts = []
                    
                    for entry in entries:
                        response = entry.get('response', '')
                        
                        # Count words (split on whitespace)
                        words = len(response.split())
                        word_counts.append(words)
                        
                        # Count tokens
                        tokens = len(response.split())  # Using word count as proxy for tokens
                        token_counts.append(tokens)
                    
                    # Store averages
                    results[run_type][model_name][task_name] = {
                        'avg_words': np.mean(word_counts),
                        'avg_tokens': np.mean(token_counts)
                    }
    
    # Calculate ratios and prepare CSV data
    csv_data = []
    
    # Per model per eval ratios
    for model in results['cot'].keys():
        for task in results['cot'][model].keys():
            if task in results['regular'][model]:
                cot_stats = results['cot'][model][task]
                reg_stats = results['regular'][model][task]
                
                word_ratio = cot_stats['avg_words'] / reg_stats['avg_words']
                token_ratio = cot_stats['avg_tokens'] / reg_stats['avg_tokens']
                
                csv_data.append({
                    'model': model,
                    'task': task,
                    'word_ratio': word_ratio,
                    'token_ratio': token_ratio,
                    'type': 'per_model_eval'
                })
    
    # Overall per model ratios (new section)
    for model in results['cot'].keys():
        cot_words = []
        cot_tokens = []
        reg_words = []
        reg_tokens = []
        
        for task in results['cot'][model].keys():
            if task in results['regular'][model]:
                cot_words.append(results['cot'][model][task]['avg_words'])
                cot_tokens.append(results['cot'][model][task]['avg_tokens'])
                reg_words.append(results['regular'][model][task]['avg_words'])
                reg_tokens.append(results['regular'][model][task]['avg_tokens'])
        
        if cot_words and reg_words:
            word_ratio = np.mean(cot_words) / np.mean(reg_words)
            token_ratio = np.mean(cot_tokens) / np.mean(reg_tokens)
            
            csv_data.append({
                'model': model,
                'task': 'ALL',
                'word_ratio': word_ratio,
                'token_ratio': token_ratio,
                'type': 'per_model'
            })
    
    # Per eval overall ratios
    for task in set().union(*[results['cot'][model].keys() for model in results['cot']]):
        cot_words = []
        cot_tokens = []
        reg_words = []
        reg_tokens = []
        
        for model in results['cot'].keys():
            if task in results['cot'][model] and task in results['regular'][model]:
                cot_words.append(results['cot'][model][task]['avg_words'])
                cot_tokens.append(results['cot'][model][task]['avg_tokens'])
                reg_words.append(results['regular'][model][task]['avg_words'])
                reg_tokens.append(results['regular'][model][task]['avg_tokens'])
        
        if cot_words and reg_words:
            word_ratio = np.mean(cot_words) / np.mean(reg_words)
            token_ratio = np.mean(cot_tokens) / np.mean(reg_tokens)
            
            csv_data.append({
                'model': 'ALL',
                'task': task,
                'word_ratio': word_ratio,
                'token_ratio': token_ratio,
                'type': 'per_eval'
            })
    
    # Overall ratio across all tasks and models
    all_cot_words = []
    all_cot_tokens = []
    all_reg_words = []
    all_reg_tokens = []
    
    for model in results['cot'].keys():
        for task in results['cot'][model].keys():
            if task in results['regular'][model]:
                all_cot_words.append(results['cot'][model][task]['avg_words'])
                all_cot_tokens.append(results['cot'][model][task]['avg_tokens'])
                all_reg_words.append(results['regular'][model][task]['avg_words'])
                all_reg_tokens.append(results['regular'][model][task]['avg_tokens'])
    
    overall_word_ratio = np.mean(all_cot_words) / np.mean(all_reg_words)
    overall_token_ratio = np.mean(all_cot_tokens) / np.mean(all_reg_tokens)
    
    csv_data.append({
        'model': 'ALL',
        'task': 'ALL',
        'word_ratio': overall_word_ratio,
        'token_ratio': overall_token_ratio,
        'type': 'overall'
    })
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('sample_ratios.csv', index=False)
    
    return df

if __name__ == "__main__":
    analyze_model_outputs()