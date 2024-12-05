# Candidate Sampling Evaluation

### 1. Average regret vs the number of tokens
To plot the relationship between the average regret and the number of reasoning tokens, use the script `agg_tokens_plot.py`. The `base_dir`  specifies the directory containing the results (JSON files), which include the metrics data for the algorithms obtained during inference.

#### Example of Data Format
The JSON files should follow a structure similar to this:

```json
"LLAMBO_CoT_500": {
    "av_regret": [
        0.7733333333333333
    ]
}
```

Here, LLAMBO_CoT_500 corresponds to a specific algorithm or configuration (e.g., Chain of Thoughts with 500 reasoning tokens), and av_regret is a list containing the average regret values for different trials.

#### Example Command
To generate the plot, run:
```bash
python3 agg_tokens_plot.py --base_dir /path/to/directory --algorithms LLAMBO LLAMBO_CoT_500 LLAMBO_CoT_1000
```

### 2. Average regret vs the number of observed points

# Discriminative Surrogate Model Evaluation
# End-to-end demonstration
