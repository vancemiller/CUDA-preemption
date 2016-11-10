#!/bin/bash

EXP=29
ITERATIONS=8
DELAY=0

echo "=== solo experiments ==="
echo "size, iterations, elapsed, average"
for i in `seq 0 $EXP`
do
  SIZE=$((1<<$i))
  ./big-kernel --size=$SIZE --iterations=$ITERATIONS --delay=$DELAY --priority=0 2>/dev/null &
  wait
done

echo "=== concurrent experiments ==="
echo "size, iterations, elapsed, average"
for i in `seq 0 $EXP`
do
  SIZE=$((1<<$i))
  ./big-kernel --size=$SIZE --iterations=$ITERATIONS --delay=$DELAY --priority=0 2>/dev/null &
  ./big-kernel --size=$SIZE --iterations=$ITERATIONS --delay=$DELAY --priority=0 2>/dev/null &
  wait
done
