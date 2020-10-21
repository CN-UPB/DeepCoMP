#!/bin/bash
# script for running multiple experiments sequentially

min_ues=$1
max_ues=$2
step_ues=$3
num_eval=1
echo Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Num eval: $num_eval

# IMPORTANT: use only 1 worker for reproducible results!
for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
  deepcomp --seed 42 --eps-length 100 --alg random --agent central --env medium --slow-ues $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-best --agent multi --env medium --slow-ues $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-all --agent multi --env medium --slow-ues $num_ues --sharing resource-fair --eval $num_eval
done
