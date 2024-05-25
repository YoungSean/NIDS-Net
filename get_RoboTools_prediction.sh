#!/bin/bash

start=1
end=24

for i in $(seq $start $end)
do
   padded_number=$(printf "%06d" $i)
   echo "Running script with test_path=datasets/RoboTools/test/$padded_number"
   python RoboTools_test_eval_gdino_FFA.py --test_path "datasets/RoboTools/test/$padded_number" --output_dir "exps/RoboTools/$padded_number"
done
