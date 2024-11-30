# Script to evaluate candidate point sampler.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="meta-llama/Meta-Llama-3.1-70B-Instruct"
EXPERIMENT_NUMBER=10

for dataset in "breast" "wine" "digits" "diabetes"
do
    for model in "RandomForest"
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sampling/evaluate_sampling.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 5 --engine $ENGINE --experiment_number $EXPERIMENT_NUMBER
        done
    done
done