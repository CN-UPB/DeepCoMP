#!/bin/bash
# script for running multiple experiments sequentially with different UE movement speeds

num_workers=$1
num_ues=$2
agent=$3
train_steps=$4
eval=$5
echo Num workers: $num_workers, Num. UEs: num_ues, Agent: $agent, Train steps: $train_steps, Eval episodes: $eval

# train with fixed episodes and test on both fixed and rand
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval $eval --fixed-rand-eval
# train with rand episodes and test on both
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval $eval --rand-train --fixed-rand-eval
