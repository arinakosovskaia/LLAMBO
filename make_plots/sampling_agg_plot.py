import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

#algorithms = ['TPE_IN', 'LLAMBO', 'RANDOM', 'TPE_MULTI']
#algorithms_all = ['TPE_IN', 'LLAMBO', 'RANDOM', 'TPE_MULTI', 'LLAMBO', 'LLAMBO_CoT_300', 'LLAMBO_CoT_500', 'LLAMBO_CoT_700', 'LLAMBO_CoT_1000']

metrics = {
    'av_regret': 'Avg Regret (↓)',
    'best_regret': 'Best Regret (↓)',
    'gen_var': 'Gen Variance (↑)',
    'll': 'Log Likelihood (↑)'
}

def extract_n_from_filename(filename):
    return int(filename.split('_')[0].split('.')[0])

def process_files_single(directory):
    results = {algo: {metric: {} for metric in metrics} for algo in algorithms}
    
    for file in os.listdir(directory):
        if file.endswith(".json"):
            n_points = extract_n_from_filename(file)
            file_path = os.path.join(directory, file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                for algo in algorithms:
                    if algo in data: 
                        for metric in metrics:
                            if metric in data[algo]:
                                values = data[algo][metric]
                                if n_points not in results[algo][metric]:
                                    results[algo][metric][n_points] = []
                                results[algo][metric][n_points].extend(values)
    
    return results

def aggregate_datasets(all_results):
    aggregated = {algo: {metric: {} for metric in metrics} for algo in algorithms}

    for dataset_results in all_results:
        for algo in algorithms:
            for metric in metrics:
                for n_points, values in dataset_results[algo][metric].items():
                    if n_points not in aggregated[algo][metric]:
                        aggregated[algo][metric][n_points] = {'means': [], 'stds': [], 'sizes': []}
                    
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    size = len(values)
                    
                    aggregated[algo][metric][n_points]['means'].append(mean)
                    aggregated[algo][metric][n_points]['stds'].append(std)
                    aggregated[algo][metric][n_points]['sizes'].append(size)

    for algo in algorithms:
        for metric in metrics:
            for n_points in aggregated[algo][metric]:
                dataset_means = np.array(aggregated[algo][metric][n_points]['means'])
                dataset_stds = np.array(aggregated[algo][metric][n_points]['stds'])
                dataset_sizes = np.array(aggregated[algo][metric][n_points]['sizes'])
                overall_mean = np.sum(dataset_means * dataset_sizes) / np.sum(dataset_sizes)

                overall_variance = (
                    np.sum((np.array(dataset_sizes) - 1) * np.array(dataset_stds)**2) + 
                    np.sum(dataset_sizes * (np.array(dataset_means) - overall_mean)**2)
                ) / (np.sum(dataset_sizes) - 1)

                aggregated[algo][metric][n_points] = {
                    'mean': overall_mean,
                    'std': np.sqrt(overall_variance)
                }
    
    return aggregated


def plot_aggregated_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, label in metrics.items():
        plt.figure(figsize=(8, 5))

        for algo in algorithms:
            x_values = sorted(results[algo][metric].keys())
            y_means = [results[algo][metric][n]['mean'] for n in x_values]
            y_stds = [results[algo][metric][n]['std'] for n in x_values]

            plt.plot(x_values, y_means, label=algo, marker='o')
            plt.fill_between(x_values, 
                             np.array(y_means) - np.array(y_stds), 
                             np.array(y_means) + np.array(y_stds), 
                             alpha=0.2)

        plt.xlabel("Number of Observed Points")
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

output_dir = os.path.join(base_dir, "aggregated_plots")
plot_aggregated_results(aggregated_results, output_dir)
