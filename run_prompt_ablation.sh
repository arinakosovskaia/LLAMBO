# Script to run ablation study on prompt design.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="meta-llama/Meta-Llama-3.1-70B-Instruct"
max_reasoning_tokens=800
prompting="cot"

for dataset in "digits"
do
    for model in "RandomForest"
    do
        for ablation_type in "full_context"
        do
            python3 exp_prompt_ablation/run_ablation.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE --ablation_type $ablation_type --shuffle_features False --max_reasoning_tokens $max_reasoning_tokens --prompting $prompting
        done
    done
done