#!/bin/bash
# script for running multiple experiments sequentially

min_ues=$1
max_ues=$2
step_ues=$3
env=$4
# num workers only relevant for brute-force
workers=$5
num_eval=30
seed=42
sharing=mixed
echo Min UEs: $min_ues, Max UEs: $max_ues, Step UEs: $step_ues, Env: $env, Sharing: $sharing, Num eval: $num_eval, Seed: $seed, Workers: $workers

for num_ues in $(seq $min_ues $step_ues $max_ues)
do
  for alg in brute-force greedy-best greedy-all
  do
    echo Num. UEs: $num_ues, Alg: $alg
    deepcomp --seed $seed --alg $alg --agent multi --env $env --slow-ues $num_ues --eval $num_eval --video html --sharing $sharing --workers $workers --rand-test
  done
  # do random separately since it's central agent
  deepcomp --seed $seed --alg random --agent central --env $env --slow-ues $num_ues --eval $num_eval --video html --sharing $sharing --rand-test
done
