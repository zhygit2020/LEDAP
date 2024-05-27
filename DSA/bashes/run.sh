#!/usr/bin/bash

export DATASOURCE="NC"
export LLAMA_SIZE="2-7b"
export FEATURE_PATH={FEATURE_PATH, here "../../representaions/" as an example}
export LABEL_PATH="../data/${DATASOURCE}_label"

python -u ../xgboost/run_xgboost.py \
    --prj_path "" \
    --label_path $LABEL_PATH \
    --feature_path $FEATURE_PATH \
    --pretrained_data_path "xgboost_${LLAMA_SIZE}_${DATASOURCE}" \
    --optuna_log_path "xgboost_${LLAMA_SIZE}_${DATASOURCE}" \
    --monitor "rmse" \
    --k_fold_num 10 \
    --epoch_num 128 \
    --patience 5 \
    --num_works 16 \
    --num_trials 4 \
    --llama_size $LLAMA_SIZE \
    --negtive_sample "none" \
    --opt_dir "minimize" \

