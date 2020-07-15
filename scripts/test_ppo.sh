#!/bin/bash
# script for loading and testing multiple trained PPO agents sequentially
# CLI args: list of PPO result dirs to test (variable length)
# eg, "PPO_CentralMultiUserEnv_0_2020-07-14_17-30-04ckm_wang PPO_CentralMultiUserEnv_0_2020-07-14_17-43-46milkk4nd"

# TODO: since each PPO test cannot be parallelized, I might run multiple tests in parallel using GNU parallel

for dir in "$@"
do
  for num_ues in {1..10}
  do
    echo Dir: "$dir", Num. UEs: $num_ues
    deepcomp --eps-length 100 --alg ppo --agent central --env medium --slow-ues $num_ues --eval 50 --test "$dir"
  done
done
