import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing evaluation results.")
parser.add_argument(
    "--algorithms",
    type=str,
    nargs="+",
    default=['LLAMBO', 'LLAMBO_CoT_500', 'LLAMBO_CoT_1000', 'LLAMBO_CoT_2000', 'LLAMBO_CoT_3000'],
    help="List of algorithms to evaluate. Default: ['LLAMBO', 'LLAMBO_CoT_500', 'LLAMBO_CoT_1000', 'LLAMBO_CoT_2000', 'LLAMBO_CoT_3000']"
)

args = parser.parse_args()
base_dir = args.base_dir
algorithms = args.algorithms

reasoning_tokens = {algo: int(algo.split('_')[-1]) if '_CoT_' in algo else 0 for algo in algorithms}

metrics = {
    'av_regret': 'Avg Regret (↓)',
    'best_regret': 'Best Regret (↓)',
    'gen_var': 'Gen Variance (↑)',
    'll': 'Log Likelihood (↑)'
}

def extract_n_from_filename(filename):
    return int(filename.split('_')[0].split('.')[0])

def process_files_single(directory):
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

def aggregate_datasets(all_results):
    aggregated = {metric: {} for metric in metrics}

    for dataset_results in all_results:
        for metric in metrics:
            for n_points, token_values in dataset_results[metric].items():
                if n_points not in aggregated[metric]:
                    aggregated[metric][n_points] = {}
                
                for tokens, values in token_values.items():
                    if tokens not in aggregated[metric][n_points]:
                        aggregated[metric][n_points][tokens] = {'means': [], 'stds': [], 'sizes': []}
                    
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    size = len(values)
                    
                    aggregated[metric][n_points][tokens]['means'].append(mean)
                    aggregated[metric][n_points][tokens]['stds'].append(std)
                    aggregated[metric][n_points][tokens]['sizes'].append(size)

    for metric in metrics:
        for n_points in aggregated[metric]:
            for tokens in aggregated[metric][n_points]:
                dataset_means = np.array(aggregated[metric][n_points][tokens]['means'])
                dataset_stds = np.array(aggregated[metric][n_points][tokens]['stds'])
                dataset_sizes = np.array(aggregated[metric][n_points][tokens]['sizes'])

                overall_mean = np.sum(dataset_means * dataset_sizes) / np.sum(dataset_sizes)

                overall_variance = (
                    np.sum((dataset_sizes - 1) * dataset_stds**2) +
                    np.sum(dataset_sizes * (dataset_means - overall_mean)**2)
                ) / (np.sum(dataset_sizes) - 1)

                aggregated[metric][n_points][tokens] = {
                    'mean': overall_mean,
                    'std': np.sqrt(overall_variance)
                }
    
    return aggregated

def aggregate_datasets(all_results):
    aggregated = {metric: {} for metric in metrics}

    for dataset_results in all_results:
        for metric in metrics:
            for n_points, token_values in dataset_results[metric].items():
                if n_points not in aggregated[metric]:
                    aggregated[metric][n_points] = {}
                
                for tokens, values in token_values.items():
                    if tokens not in aggregated[metric][n_points]:
                        aggregated[metric][n_points][tokens] = {'means': [], 'stds': [], 'sizes': []}
                    
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    size = len(values)
                    
                    aggregated[metric][n_points][tokens]['means'].append(mean)
                    aggregated[metric][n_points][tokens]['stds'].append(std)
                    aggregated[metric][n_points][tokens]['sizes'].append(size)

    for metric in metrics:
        for n_points in aggregated[metric]:
            for tokens in aggregated[metric][n_points]:
                dataset_means = np.array(aggregated[metric][n_points][tokens]['means'])
                dataset_stds = np.array(aggregated[metric][n_points][tokens]['stds'])
                dataset_sizes = np.array(aggregated[metric][n_points][tokens]['sizes'])

                overall_mean = np.sum(dataset_means * dataset_sizes) / np.sum(dataset_sizes)

                overall_variance = (
                    np.sum((dataset_sizes - 1) * dataset_stds**2) +
                    np.sum(dataset_sizes * (dataset_means - overall_mean)**2)
                ) / (np.sum(dataset_sizes) - 1)

                aggregated[metric][n_points][tokens] = {
                    'mean': overall_mean,
                    'std': np.sqrt(overall_variance)
                }
    
    return aggregated

def plot_aggregated_reasoning_vs_metric(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, label in metrics.items():
        plt.figure(figsize=(8, 5))
        
        for n_points in sorted(results[metric].keys()):
            tokens = sorted(results[metric][n_points].keys())
            y_means = [results[metric][n_points][t]['mean'] for t in tokens]
            y_stds = [results[metric][n_points][t]['std'] for t in tokens]
            
            plt.plot(tokens, y_means, label=f"n={n_points}", marker='o')
            plt.fill_between(tokens, 
                             np.array(y_means) - np.array(y_stds), 
                             np.array(y_means) + np.array(y_stds), 
                             alpha=0.2)
        
        plt.xlabel("Reasoning Tokens")
        plt.ylabel(label)
        plt.title(f"Averaged {label} Across Datasets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"averaged_{metric}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"averaged_{metric}.pdf"))
        plt.close()

all_results = []

for dataset in os.listdir(base_dir):
    dataset_dir = os.path.join(base_dir, dataset, "RandomForest")  
    if os.path.isdir(dataset_dir):
        print(f"Processing dataset: {dataset}")
        dataset_results = process_files_single(dataset_dir)
        all_results.append(dataset_results)

aggregated_results = aggregate_datasets(all_results)

output_dir = os.path.join(base_dir, "aggregated_reasoning_vs_metric_plots")
plot_aggregated_reasoning_vs_metric(aggregated_results, output_dir)
