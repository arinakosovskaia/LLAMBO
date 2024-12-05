import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing evaluation results.")
parser.add_argument(
    "--algorithms",
    type=str,
    nargs="+",
    default=['LLAMBO', 'LLAMBO_CoT_300', 'LLAMBO_CoT_500', 'LLAMBO_CoT_700', 'LLAMBO_CoT_1000'],
    help="List of algorithms to evaluate. Default: ['LLAMBO', 'LLAMBO_CoT_300', 'LLAMBO_CoT_500', 'LLAMBO_CoT_700', 'LLAMBO_CoT_1000']"
)
args = parser.parse_args()

base_dir = args.base_dir
algorithms = args.algorithms

# Define reasoning tokens for each algorithm
reasoning_tokens = {
    'LLAMBO': 0,
    'LLAMBO_CoT_500': 500,
    'LLAMBO_CoT_1000': 1000,
    'LLAMBO_CoT_2000': 2000,
    'LLAMBO_CoT_3000': 3000,
}

metrics = {
    'av_regret': 'Avg Regret (↓)',
    'best_regret': 'Best Regret (↓)',
    'gen_var': 'Gen Variance (↑)',
    'll': 'Log Likelihood (↑)'
}

def extract_n_from_filename(filename):
    return int(filename.split('_')[0].split('.')[0])

def process_files(directory):
    results = {metric: {} for metric in metrics}
    
    for file in os.listdir(directory):
        if file.endswith(".json"):
            n_points = extract_n_from_filename(file)
            file_path = os.path.join(directory, file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                for algo, tokens in reasoning_tokens.items():
                    if algo in data:
                        for metric in metrics:
                            if metric in data[algo]:
                                values = data[algo][metric]
                                if n_points not in results[metric]:
                                    results[metric][n_points] = {}
                                if tokens not in results[metric][n_points]:
                                    results[metric][n_points][tokens] = []
                                results[metric][n_points][tokens].extend(values)
    
    return results

def plot_reasoning_vs_metric(dataset_name, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, label in metrics.items():
        plt.figure(figsize=(8, 5))
        
        for n_points in sorted(results[metric].keys()):
            tokens = sorted(results[metric][n_points].keys())
            y_means = [np.mean(results[metric][n_points][t]) for t in tokens]
            y_stds = [np.std(results[metric][n_points][t]) for t in tokens]
            
            plt.plot(tokens, y_means, label=f"n={n_points}", marker='o')
            plt.fill_between(tokens, 
                             np.array(y_means) - np.array(y_stds), 
                             np.array(y_means) + np.array(y_stds), 
                             alpha=0.2)
        
        plt.xlabel("Reasoning Tokens")
        plt.ylabel(label)
        plt.title(f"{label} vs Reasoning Tokens on {dataset_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{metric}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{metric}.pdf"))
        plt.close()

for dataset in os.listdir(base_dir):
    dataset_dir = os.path.join(base_dir, dataset, "RandomForest")  
    if os.path.isdir(dataset_dir):
        print(f"Processing dataset: {dataset}")
        results = process_files(dataset_dir) 
        output_dir = os.path.join(base_dir, dataset, "plots_reasoning_vs_metric") 
        plot_reasoning_vs_metric(dataset, results, output_dir)

