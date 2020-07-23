#!/bin/bash
# script for loading and testing multiple trained PPO agents sequentially
# CLI args: list of PPO result dirs to test (variable length); eg, './test_ppo.sh ../results/PPO/PPO*'
# eg, "PPO_CentralMultiUserEnv_0_2020-07-14_17-30-04ckm_wang PPO_CentralMultiUserEnv_0_2020-07-14_17-43-46milkk4nd"

# TODO: since each PPO test cannot be parallelized, I might run multiple tests in parallel using GNU parallel
#  not sure if I can run multiple ray sessions in parallel? may break

# load and test saved PPO agents in order; assuming they were for 1-10 agents respectively
min_ues=10
max_ues=40
step_ues=10

for dir in "$@"
do
  for num_ues in $(seq $min_ues $step_ues $max_ues)
  do
    echo Dir: "$dir", Num. UEs: $num_ues, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues
    deepcomp --seed 42 --eps-length 100 --alg ppo --agent multi --env medium --slow-ues $num_ues --sharing proportional-fair --eval 100 --test "$dir"
  done
done
