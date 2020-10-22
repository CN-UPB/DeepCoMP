#!/bin/bash
# script for running multiple experiments sequentially

num_workers=$1
min_ues=$2
max_ues=$3
step_ues=$4
agent=$5
train_steps=$6
echo Num workers: $num_workers, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Agent: $agent, Train steps: $train_steps

for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env medium --slow-ues $num_ues --sharing resource-fair --eval 1
done
