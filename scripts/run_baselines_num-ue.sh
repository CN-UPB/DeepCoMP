#!/bin/bash
# script for running multiple experiments sequentially

min_ues=$1
max_ues=$2
step_ues=$3
env=$4
num_eval=1
echo Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Env: $env, Num eval: $num_eval

# IMPORTANT: use only 1 worker for reproducible results!
for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  echo Num. UEs: $num_ues
#  deepcomp --seed 42 --alg random --agent central --env $env --slow-ues $num_ues --eval $num_eval --video html
  deepcomp --seed 42 --alg greedy-best --agent multi --env $env --slow-ues $num_ues --eval $num_eval --video html
  deepcomp --seed 42 --alg greedy-all --agent multi --env $env --slow-ues $num_ues --eval $num_eval --video html
  # run brute force once optimizing sum utility and once optimizing min utility
  deepcomp --seed 42 --alg brute-force --agent central --env $env --slow-ues $num_ues --eval $num_eval --video html --reward sum
  deepcomp --seed 42 --alg brute-force --agent central --env $env --slow-ues $num_ues --eval $num_eval --video html --reward min
done
