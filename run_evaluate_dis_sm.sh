# Script to evaluate discriminative surrogate model.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="mistralai/Mixtral-8x22B-Instruct-v0.1"

for dataset in "wine" "digits" "diabetes" "breast" 
do
    for model in "RandomForest" 
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sm/evaluate_dis_sm.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 3 --engine $ENGINE
        done
    done
done