#!/bin/bash

JSON_FILE="data/hpob-data/meta-test-dataset.json"
start=0
MAX_PROCESSES=1
current_processes=0

declare -A dim_task=(
    [2]="5860 5970"
    [6]="5859 5889"
    [9]="7607 7609"
    [16]="5906 5971"
)

for dim in "${!dim_task[@]}"; do
    read -ra ids <<< "${dim_task[$dim]}"
    for search_space_id in "${ids[@]}"; do
        echo "dim: $dim, search_space_id: $search_space_id"
        dataset_ids=$(jq -r --arg ss_id "$search_space_id" '.[$ss_id] | keys[]' "$JSON_FILE")
        for dataset_id in $dataset_ids; do
            data_count=$(jq -r --arg ss_id "$search_space_id" --arg ds_id "$dataset_id" '.[$ss_id][$ds_id].X | length' "$JSON_FILE")
            echo "Data count=$data_count for search_space_id=$search_space_id dataset_id=$dataset_id"
            if [ "$data_count" -gt 30000 ]; then
                echo "Data count=$data_count for search_space_id=$search_space_id dataset_id=$dataset_id is greater than 30000, skipping."
                continue
            fi
            args="--mode=hpob --search-space-id=$search_space_id --dataset-id=$dataset_id --iteration=100 --similar=combine --weight-update=linear-half --similarity=topN --N=5.0 --Cp=0.1"
            echo "Running python run.py with args: $args"
            python run.py $args &
            ((current_processes++))
            if [[ current_processes -ge MAX_PROCESSES ]]; then
                wait -n
                ((current_processes--))
            fi
        done
    done
done