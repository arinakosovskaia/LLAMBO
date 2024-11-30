import argparse
import os
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Process file paths and output directory.")
parser.add_argument('file_paths', nargs='+', help='Paths to the JSON result files')
parser.add_argument('output_dir', help='Directory to save the output plot files')
parser.add_argument('save_file_name', help='Path to the file where plots will be saved')
args = parser.parse_args()

results = {
    "cot_300": {},
    "cot_500": {},
    "zero_shot": {}
}

labels = ['cot_300', 'cot_500', 'zero_shot']
keys = ['regrets', 'best_f_val']

for idx, file_path in enumerate(args.file_paths):
    with open(file_path, 'r') as file:
        data = json.load(file)

    regrets = data["regrets"][0]  
    best_f_val = data["best_f_val"]

    if idx == 0:
        results["cot_300"]["regrets"] = regrets
        results["cot_300"]["best_f_val"] = best_f_val
    elif idx == 1:
        results["cot_500"]["regrets"] = regrets
        results["cot_500"]["best_f_val"] = best_f_val
    else:
        results["zero_shot"]["regrets"] = regrets
        results["zero_shot"]["best_f_val"] = best_f_val

for label in labels:
    print(f"Best f_val for {label}: {results[label]['best_f_val']}")

plt.figure(figsize=(10, 6))

for label in labels:
    plt.plot(results[label]["regrets"], label=label)

plt.xlabel('Trial')
plt.ylabel('Regrets')
plt.title('Regret on Digits Dataset')
plt.legend()
plt.grid(True)
plt.show()
plt.grid(True)
plt.savefig(os.path.join(args.output_dir, f"{args.save_file_name}.png"))
plt.savefig(os.path.join(args.output_dir, f"{args.save_file_name}.pdf"))
plt.close()
