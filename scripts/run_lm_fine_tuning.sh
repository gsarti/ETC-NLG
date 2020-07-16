#!/bin/bash
 
############
# settings #
############

MODEL="Svevo"
DATASET="gold" # gold, contextual, combined
BLOCK_SIZE=128
LM_EPOCHS=5

################
# fine-tune LM #
################

TESTS="../tests/${MODEL}_${DATASET}"
export TRAIN_FILE="${TESTS}/${MODEL}_${DATASET}_train.txt"
export TEST_FILE="${TESTS}/${MODEL}_${DATASET}_test.txt"
OUT_DIR="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
OUT="${OUT_DIR}/lm_fine_tuning_out.txt"

source ../venv/bin/activate
mkdir -p $OUT_DIR

python3 transformers/run_language_modeling.py \
    --output_dir=$OUT_DIR --cache_dir="../tests/cache"\
    --model_type=gpt2 --model_name_or_path="LorenzoDeMattei/GePpeTto" \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval --eval_data_file=$TEST_FILE \
    --block_size=$BLOCK_SIZE --save_steps=1000 --save_total_limit=1 \
    --overwrite_output_dir --epochs=$LM_EPOCHS > $OUT 
    #--no_cuda 

deactivate