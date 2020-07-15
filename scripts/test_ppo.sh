#!/bin/bash
# script for loading and testing multiple trained PPO agents sequentially
# CLI args: list of PPO result dirs to test (variable length); eg, './test_ppo.sh ../results/PPO/PPO*'
# eg, "PPO_CentralMultiUserEnv_0_2020-07-14_17-30-04ckm_wang PPO_CentralMultiUserEnv_0_2020-07-14_17-43-46milkk4nd"

# TODO: since each PPO test cannot be parallelized, I might run multiple tests in parallel using GNU parallel
#  not sure if I can run multiple ray sessions in parallel? may break

# load and test saved PPO agents in order; assuming they were for 1-10 agents respectively
num_ues=1
for dir in "$@"
do
  echo Dir: "$dir", Num. UEs: $num_ues
  deepcomp --eps-length 100 --alg ppo --agent central --env medium --slow-ues $num_ues --eval 50 --test "$dir"
  num_ues=$((num_ues + 1))
done
