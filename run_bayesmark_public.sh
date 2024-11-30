# Script to run LLAMBO on all Bayesmark tasks.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
#ENGINE="meta-llama/Meta-Llama-3.1-70B-Instruct"

ENGINE="gpt-3.5-turbo-0125"
ENGINE='meta-llama/Meta-Llama-3.1-70B-Instruct'
for dataset in "digits"
do
    for model in "RandomForest"
    do
        python3 exp_bayesmark/run_bayesmark.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE
        sleep 60
    done
done