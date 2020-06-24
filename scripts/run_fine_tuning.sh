#!/bin/bash

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests"
OUT_DIR="${TESTS}/${DATE}/geppetto_svevo_blockSize=${BLOCK_SIZE}"
mkdir -p $TESTS

############
# settings #
############

export TRAIN_FILE="../data/letters_train.txt"
export TEST_FILE="../data/letters_test.txt"
BLOCK_SIZE=100

#########
# train #
#########

python3 copyrighted_code/run_language_modeling.py \
    --output_dir=$OUT_DIR \
    --cache_dir="${TESTS}/cache"\
    --model_type=gpt2 \
    --model_name_or_path="LorenzoDeMattei/GePpeTto" \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size=$BLOCK_SIZE 

deactivate