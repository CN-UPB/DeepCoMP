#!/bin/bash
# script for running multiple experiments sequentially with different UE movement speeds

num_workers=$1
num_ues=$2
agent=$3
train_steps=$4
echo Num workers: $num_workers, Num. UEs: num_ues, Agent: $agent, Train steps: $train_steps

deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval 1 --video html --sharing resource-fair
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval 1 --video html --sharing rate-fair
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval 1 --video html --sharing max-cap
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval 1 --video html --sharing proportional-fair
deepcomp --seed 42 --workers $num_workers --train-steps $train_steps --agent $agent --env medium --slow-ues $num_ues --eval 1 --video html --sharing mixed
