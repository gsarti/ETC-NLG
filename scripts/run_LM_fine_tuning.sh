#!/bin/bash

############
# settings #
############

export TRAIN_FILE="../data/letters_train.txt"
export TEST_FILE="../data/letters_test.txt"
BLOCK_SIZE=512
LM_EPOCHS=5
SAVE_DIR="Svevo"

#########
# train #
#########

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${SAVE_DIR}"
mkdir -p $TESTS
OUT_DIR="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
OUT="${OUT_DIR}/lm_fine_tuning_out.txt"

python3 transformers/run_language_modeling.py \
    --output_dir=$OUT_DIR --cache_dir="../tests/cache"\
    --model_type=gpt2 --model_name_or_path="LorenzoDeMattei/GePpeTto" \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval --eval_data_file=$TEST_FILE \
    --block_size=$BLOCK_SIZE --save_steps=1000 --save_total_limit=1 \
    --overwrite_output_dir --epochs=$LM_EPOCHS > $OUT #--no_cuda 

deactivate