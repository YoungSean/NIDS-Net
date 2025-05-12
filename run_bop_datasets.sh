#!/bin/bash

# List of datasets
datasets=("lmo" "tless" "tudl" "icbin" "itodd" "hb" "ycbv")

# Loop through each dataset
for DATASET_NAME in "${datasets[@]}"; do
    export DATASET_NAME=$DATASET_NAME
    echo "Running inference on dataset: $DATASET_NAME"
    python run_inference.py dataset_name=$DATASET_NAME
done
