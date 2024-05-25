#!/bin/bash

start=48
end=59

for i in $(seq $start $end)
do
   padded_number=$(printf "%06d" $i)
   echo "Running script with test_path=datasets/ycbv/test/$padded_number"
   python ycbv_test_eval_gdino_FFA.py --test_path "datasets/ycbv/test/$padded_number" --output_dir "exps/ycbv/$padded_number"
done
