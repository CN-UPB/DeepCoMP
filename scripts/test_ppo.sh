#!/bin/bash
# script for loading and testing multiple trained PPO agents sequentially
# CLI args: list of PPO result dirs to test (variable length); eg, './test_ppo.sh ../results/PPO/PPO*'
# or: "PPO_CentralMultiUserEnv_0_2020-07-14_17-30-04ckm_wang PPO_CentralMultiUserEnv_0_2020-07-14_17-43-46milkk4nd"

# load and test saved PPO agents in order; using the num UEs specified below
min_ues=1
max_ues=5
step_ues=1

for dir in "$@"
do
  for num_ues in $(seq $min_ues $step_ues $max_ues)
  do
    echo Dir: "$dir", Num. UEs: $num_ues, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues
    deepcomp --seed 42 --eps-length 100 --alg ppo --agent multi --env medium --slow-ues $num_ues --sharing proportional-fair --eval 5 --test "$dir"
  done
done
