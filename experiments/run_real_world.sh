#!/bin/bash

start=0
MAX_PROCESSES=1
current_processes=0

search_space_ids=("LunarLander" "RobotPush" "Rover")
dataset_ids=("0")

for search_space_id in "${search_space_ids[@]}"; do
    for dataset_id in "${dataset_ids[@]}"; do
        args="--mode=real_world --search-space-id=$search_space_id --dataset-id=$dataset_id --iteration=100 --similar=mix-similar --weight-update=linear-half --kernel-type=lr --Cp=0.1 --similarity=topN --N=5.0"
        echo "Running python run.py with args: $args"
        python run.py $args &
        ((current_processes++))
        if [[ current_processes -ge MAX_PROCESSES ]]; then
            wait -n
            ((current_processes--))
        fi
    done
done