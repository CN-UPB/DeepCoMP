#!/bin/bash
# script for running multiple experiments sequentially

for num_ues in {1..10}
do
  echo Num. UEs: $num_ues
#  deepcomp --workers 2 --eps-length 100 --alg greedy-best --agent multi --env medium --slow-ues $num_ues --eval 50
  deepcomp --workers 15 --eps-length 100 --train-iter 100 --batch-size 4000 --alg ppo --agent multi --env medium --slow-ues $num_ues --eval 50
done
