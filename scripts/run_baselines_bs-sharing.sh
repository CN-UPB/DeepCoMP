#!/bin/bash
# script for running multiple experiments sequentially

num_ues=$1
workers=$2
num_eval=1
echo Num UEs: $num_ues, Num eval: $num_eval, Workers: $workers

# IMPORTANT: use only 1 worker for reproducible results!
for sharing in resource-fair rate-fair max-cap proportional-fair mixed
do
  for alg in brute-force
  do
    echo Sharing: $sharing, Alg: $alg
    deepcomp --seed 42 --alg $alg --env medium --slow-ues $num_ues --eval $num_eval --video html --sharing $sharing --workers $workers
  done

done
