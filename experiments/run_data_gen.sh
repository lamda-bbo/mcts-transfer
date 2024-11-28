#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

max_cpu=$(cat /proc/cpuinfo | grep "processor" | wc -l)
echo 'max_cpu: '$max_cpu
max_proc=12
n_cpu=`expr $max_cpu / $max_proc`
echo 'num cpu per process: '$n_cpu

min_seed=0
max_seed=3
fifo_name="/tmp/$$.fifo"
mkfifo $fifo_name
exec 7<>${fifo_name}
# rm $fifo_name

for ((i=0; i<$max_proc; i++))
do
    echo $i
done >&7

for ((seed=${min_seed}; seed<=${max_seed}; seed++))
do
    read -u7 proc_id
    echo 'current proc id: '$proc_id
    {
        cpu_start=`expr $proc_id \* $n_cpu`
        cpu_end=`expr \( $proc_id + 1 \) \* $n_cpu - 1`
        python run_data_gen.py \
            --problem=synthetic \
            --seed=$seed \
            --cpu_start=$cpu_start \
            --cpu_end=$cpu_end \
            --mode=train
        sleep 1
        echo >&7 $proc_id
    } &
done

wait
exec 7>&-