#!/bin/bash

make clean && make acocpuomp -j4

tspBase="tsp/"

tspArray=(
    "bays29.tsp"
    "d198.tsp"
    "pcb442.tsp"
    "pr1002.tsp"
    "rl1889.tsp"
    "d2103.tsp"
    "pcb3038.tsp"
)


for problem in "${tspArray[@]}"
do
#   ./acocpuomp file.tsp         alpha   beta     q   rho maxEpoch mode
    ./acocpuomp $tspBase$problem   1       2       1   0.5 10       0
    ./acocpuomp $tspBase$problem   1       2       1   0.5 10       1
done