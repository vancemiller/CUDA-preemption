#!/bin/bash

if [ "$#" -lt "2" ]
then
  echo "Please specify two arguments: number of iterations and number of processes."
  echo "Example: launch_n_processes.sh 2048 4"
  exit
fi
N_ITERATIONS=$1
N_PROCESSES=$((${2} / 2))
DATA_SIZE=$((1024*1024))


for i in `seq 1 $N_PROCESSES`;
do
  ./one-stream --iterations=${N_ITERATIONS} --size=${DATA_SIZE} --priority=-1 &
  pids[${i}]=$!
done

for i in `seq 1 $N_PROCESSES`;
do
  ./one-stream --iterations=${N_ITERATIONS} --size=${DATA_SIZE} --priority=0 --delay &
  pids[$((${N_PROCESSES}+${i}))]=$!
done

for i in `seq 1 $((2 * ${N_PROCESSES}))`;
do
  wait ${pids[$i]}
done

