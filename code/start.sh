#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

seed_start=2018
seed_end=2022

func_list=(Lasso)
max_samples=1000
sample_batch_size=1
root_dir=logs
device="cuda:3"
Cp=0.1

for func in ${func_list[@]}
do
    max_group_size=10
    for ((seed=$seed_start; seed<=$seed_end; seed++)); do

        python3 additive_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --sample_batch_size=$sample_batch_size \
            --grouping_strategy=CapacitatedKmeans \
            --metric_type=single \
            --metric_classes=RegressionInteractionScore \
            --max_group_size=$max_group_size \
            --root_dir=$root_dir \
            --seed=$seed \
            --device=$device \
            --acq_opt_mode=global \
            --acq_opt_method=turbo \
            --precision_record=yes

        python3 additive_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --sample_batch_size=$sample_batch_size \
            --grouping_strategy=RTree \
            --max_group_size=$max_group_size \
            --root_dir=$root_dir \
            --seed=$seed \
            --device=$device \
            --acq_opt_mode=global \
            --acq_opt_method=turbo \
            --precision_record=yes

        python3 additive_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --sample_batch_size=$sample_batch_size \
            --grouping_strategy=Tree \
            --max_group_size=$max_group_size \
            --root_dir=$root_dir \
            --seed=$seed \
            --device=$device \
            --acq_opt_mode=global \
            --acq_opt_method=turbo \
            --precision_record=yes


        python3 baseline/turbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed

    done

done