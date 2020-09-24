#!/bin/bash
# script for running multiple, custom experiments sequentially
# pass central or multi as CLI arg, configure rest in script

agent=$1
num_workers=14
env=large
trainsteps=500000

deepcomp --seed 42 --workers $num_workers --train-steps $trainsteps --agent $agent --env $env --slow-ues 2
deepcomp --seed 42 --workers $num_workers --train-steps $trainsteps --agent $agent --env $env --slow-ues 7
deepcomp --seed 42 --workers $num_workers --train-steps $trainsteps --agent $agent --env $env --fast-ues 7
