#!/bin/bash
# script for running multiple experiments sequentially

num_ues=$1
num_eval=1
echo Num UEs: $num_ues, Num eval: $num_eval

# IMPORTANT: use only 1 worker for reproducible results!
for sharing in resource-fair rate-fair max-cap proportional-fair
do
  for alg in greedy-best greedy-all
  do
    echo Sharing: $sharing, Alg: $alg
    deepcomp --seed 42 --alg $alg --agent multi --env medium --$movement $num_ues --eval $num_eval --video html --sharing $sharing
  done
done
