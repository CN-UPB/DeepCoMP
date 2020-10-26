#!/bin/bash
# script for running multiple experiments sequentially

num_workers=$1
min_ues=$2
max_ues=$3
step_ues=$4
env=$5
agent=$6
train_steps=$7
echo Num workers: $num_workers, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Env: $env, Agent: $agent, Train steps: $train_steps

for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env $env --slow-ues $num_ues --eval 1 --video html
done
