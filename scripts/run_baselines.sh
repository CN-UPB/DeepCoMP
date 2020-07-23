#!/bin/bash
# script for running multiple experiments sequentially
# CLI args: num_workers, min_ues, max_ues

# TODO: use only 1 worker for reproducible results!
num_workers=$1
min_ues=$2
max_ues=$3
step_ues=$4
echo Num workers: $num_workers, Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues

for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
#  deepcomp --seed 42 --workers $num_workers --eps-length 100 --alg random --agent central --env medium --slow-ues $num_ues --sharing proportional-fair --eval 100
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --alg greedy-best --agent multi --env medium --slow-ues $num_ues --sharing proportional-fair --eval 100
  deepcomp --seed 42 --workers $num_workers --eps-length 100 --alg greedy-all --agent multi --env medium --slow-ues $num_ues --sharing proportional-fair --eval 100
done
