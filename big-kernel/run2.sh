#!/bin/bash

SIZE=$((1024 * 40)) ## 2048 * 20 ## number of sms time threads/sm
ITERATIONS=1
DELAY=$((1024))
N_PROCESSES=1
nice ./big-kernel --size=$SIZE --iterations=$ITERATIONS --delay=$DELAY --priority=0 --processes=$N_PROCESSES &
./big-kernel --size=$SIZE --iterations=$ITERATIONS --delay=$DELAY --priority=-1 --processes=$N_PROCESSES &
wait
