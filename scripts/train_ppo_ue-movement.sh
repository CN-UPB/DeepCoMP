#!/bin/bash
# script for running multiple experiments sequentially with different UE movement speeds

num_workers=$1
num_ues=$2
agent=$3
train_steps=$4
echo Num workers: $num_workers, Num. UEs: num_ues, Agent: $agent, Train steps: $train_steps

deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env medium --static-ues $num_ues --sharing resource-fair --eval 1
deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env medium --slow-ues $num_ues --sharing resource-fair --eval 1
deepcomp --seed 42 --workers $num_workers --eps-length 100 --train-steps $train_steps --alg ppo --agent $agent --env medium --fast-ues $num_ues --sharing resource-fair --eval 1
