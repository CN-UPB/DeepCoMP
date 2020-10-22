#!/bin/bash
# script for running multiple experiments sequentially

min_dist=$1
max_dist=$2
step_dist=$3
num_ues=3
num_eval=1
echo Num UEs: $num_ues, Min dist: $min_dist, Max dist: $max_dist, Step dist: $step_dist

# IMPORTANT: use only 1 worker for reproducible results!
for dist in $(seq $min_dist $step_dist $max_dist)
do
  echo Dist: $dist
  deepcomp --seed 42 --eps-length 100 --alg random --agent central --env medium --bs-dist $dist --slow-ues $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-best --agent multi --env medium --bs-dist $dist --slow-ues $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-all --agent multi --env medium --bs-dist $dist --slow-ues $num_ues --sharing resource-fair --eval $num_eval
done
