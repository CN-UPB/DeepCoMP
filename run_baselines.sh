#!/bin/bash
# script for running multiple experiments sequentially
# CLI args: num_workers, min_ues, max_ues

for num_ues in {$2..$3}
do
  echo Num. UEs: $num_ues
  deepcomp --workers $1 --eps-length 100 --alg random --agent central --env medium --slow-ues $num_ues --eval 50
  deepcomp --workers $1 --eps-length 100 --alg greedy-best --agent multi --env medium --slow-ues $num_ues --eval 50
  deepcomp --workers $1 --eps-length 100 --alg greedy-all --agent multi --env medium --slow-ues $num_ues --eval 50
done
