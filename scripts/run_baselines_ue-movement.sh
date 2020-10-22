#!/bin/bash
# script for running multiple experiments sequentially

num_ues=$1
num_eval=1
echo Num UEs: $num_ues, Num eval: $num_eval

# IMPORTANT: use only 1 worker for reproducible results!
for movement in static-ues slow-ues fast-ues
do
  echo UEs: --$movement $num_ues
  deepcomp --seed 42 --eps-length 100 --alg random --agent central --env medium --$movement $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-best --agent multi --env medium --$movement $num_ues --sharing resource-fair --eval $num_eval
  deepcomp --seed 42 --eps-length 100 --alg greedy-all --agent multi --env medium --$movement $num_ues --sharing resource-fair --eval $num_eval
done
