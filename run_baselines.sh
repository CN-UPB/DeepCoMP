#!/bin/bash
# script for running multiple experiments sequentially
# pass num workers as 1st cli arg

for num_ues in {1..2}
do
  echo Num. UEs: $num_ues
  deepcomp --workers $1 --eps-length 100 --alg random --agent central --env medium --slow-ues $num_ues --eval 50
  deepcomp --workers $1 --eps-length 100 --alg greedy-best --agent multi --env medium --slow-ues $num_ues --eval 50
  deepcomp --workers $1 --eps-length 100 --alg greedy-all --agent multi --env medium --slow-ues $num_ues --eval 50
done
