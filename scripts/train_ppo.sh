#!/bin/bash
# script for running multiple experiments sequentially

num_workers=$1
min_ues=$2
max_ues=$3
step_ues=$4
agent=$5
echo Num workers: $num_workers, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Agent: $agent

for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps 2000000 --batch-size 4000 --alg ppo --agent $agent --env medium --slow-ues $num_ues --sharing resource-fair --eval 100
done
