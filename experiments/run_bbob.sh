#!/bin/bash
start=0

MAX_PROCESSES=1
current_processes=0

search_space_ids=("GriewankRosenbrock" "Lunacek" "Rastrigin" "RosenbrockRotated" "SharpRidge")
dataset_ids=("0")
echo "${search_space_ids[@]}"
for search_space_id in "${search_space_ids[@]}"; do
    for dataset_id in "${dataset_ids[@]}"; do
        args="--mode=bbob --search-space-id=$search_space_id --dataset-id=$dataset_id --iteration=100 --similar=combine --weight-update=linear-half --similarity=topN --N=5.0 --Cp=0.1"
        echo "Running python run.py with args: $args"
        python run.py $args &
        ((current_processes++))
        if [[ current_processes -ge MAX_PROCESSES ]]; then
            wait -n
            ((current_processes--))
        fi
    done
done