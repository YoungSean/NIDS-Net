python -m src.scripts.inference_custom \
    --template_dir $OUTPUT_DIR \
    --rgb_path $RGB_PATH \
    --stability_score_thresh 0.3 \
    --confg_threshold 0.1 \
    --num_max_dets 1