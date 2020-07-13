#!/bin/bash

############
# settings #
############

export TRAIN_FILE="../data/letters_train.txt"
export TEST_FILE="../data/letters_test.txt"
BLOCK_SIZE=256
SAVE_DIR="Svevo"

#########
# train #
#########

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${SAVE_DIR}"
mkdir -p $TESTS
OUT_DIR="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}"

python3 transformers/run_language_modeling.py \
    --output_dir=$OUT_DIR --cache_dir="../tests/cache"\
    --model_type=gpt2 --model_name_or_path="LorenzoDeMattei/GePpeTto" \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval --eval_data_file=$TEST_FILE \
    --block_size=$BLOCK_SIZE \
    --overwrite_output_dir

deactivate