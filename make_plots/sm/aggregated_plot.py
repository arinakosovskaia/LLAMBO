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
    default=['TPE_IN', 'LLAMBO', 'RANDOM', 'TPE_MULTI', 'LLAMBO_CoT'],
    help="List of algorithms to evaluate. Default: ['TPE_IN', 'LLAMBO', 'RANDOM', 'TPE_MULTI', 'LLAMBO_CoT']"
)
args = parser.parse_args()

base_dir = args.base_dir
algorithms = args.algorithms

metrics = {
    'av_regret': 'Avg Regret (↓)',
    'best_regret': 'Best Regret (↓)',
    'gen_var': 'Gen Variance (↑)',
    'll': 'Log Likelihood (↑)'
}

def extract_n_from_filename(filename):
    return int(filename.split('_')[0].split('.')[0])

def process_files(directory, results):
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
                                value = data[algo][metric][0] 
                                if n_points not in results[algo][metric]:
                                    results[algo][metric][n_points] = []
                                results[algo][metric][n_points].append(value)

def plot_aggregated_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, label in metrics.items():
        plt.figure(figsize=(8, 5))

        for algo in algorithms:
            x_values = sorted(results[algo][metric].keys()) 
            y_means = [np.mean(results[algo][metric][n]) for n in x_values] 
            y_stds = [np.std(results[algo][metric][n]) for n in x_values] 

            # Линия и закрашенная область
            plt.plot(x_values, y_means, label=algo, marker='o')
            plt.fill_between(x_values, 
                             np.array(y_means) - np.array(y_stds), 
                             np.array(y_means) + np.array(y_stds), 
                             alpha=0.2)

        # Оформление графика
        plt.xlabel("Number of Observed Points")
        plt.ylabel(label)
        plt.title(f"Averaged {label} Across All Datasets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Сохранение графика
        plt.savefig(os.path.join(output_dir, f"aggregated_{metric}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"aggregated_{metric}.pdf"))
        plt.close()

aggregated_results = {algo: {metric: {} for metric in metrics} for algo in algorithms}

for dataset in os.listdir(base_dir):
    dataset_dir = os.path.join(base_dir, dataset, "RandomForest")
    if os.path.isdir(dataset_dir):
        print(f"Processing dataset: {dataset}")
        process_files(dataset_dir, aggregated_results)

plot_aggregated_results(aggregated_results, os.path.join(base_dir, "aggregated_plots"))

