#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# Creating initial points
for config in  "random", "sobol", "lhs", "Full_Context", "CoT"
do
    python3 exp_warmstarting/init_warmstart.py --config $config
    sleep 60
done

# It assumes  "random", "sobol", "lhs", "Full_Context", "CoT"
python3 exp_warmstarting/run_warmstart.py