#!/usr/bin/bash

export DATASOURCE="hetionet"
export LLAMA_SIZE="2-7b"
export FEATURE_PATH={FEATURE_PATH, here "../../representaions/" as an example}
export LABEL_PATH="../data/${DATASOURCE}_label"

python -u ../rf/run_rf.py \
    --prj_path "" \
    --label_path $LABEL_PATH \
    --feature_path $FEATURE_PATH \
    --pretrained_data_path "rf_${LLAMA_SIZE}_${DATASOURCE}" \
    --optuna_log_path "rf_${LLAMA_SIZE}_${DATASOURCE}" \
    --monitor "auroc" \
    --k_fold_num 10 \
    --epoch_num 128 \
    --patience 5 \
    --num_works 16 \
    --num_trials 128 \
    --llama_size $LLAMA_SIZE \
    --negtive_sample "all" \
    --opt_dir "maximize" \

