#!/bin/bash
# script for running multiple experiments sequentially with varying BS distance
# eg: 10 60 120 20 multi 100000 --> 10 workers, 60-120 BS dist (in steps of 20), PPO multi, 100k train

num_workers=$1
min_dist=$2
max_dist=$3
step_dist=$4
agent=$5
train_steps=$6
num_ues=3
echo Num workers: $num_workers, Num UEs: $num_ues, Min dist: $min_dist, Max dist: $max_dist, Step dist: $step_dist, Agent: $agent, Train steps: $train_steps

for dist in $(seq $min_dist $step_dist $max_dist)
do
  echo Dist: $dist
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env medium --bs-dist $dist --slow-ues $num_ues --sharing resource-fair --eval 1
done
