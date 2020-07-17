#!/bin/bash
 
############
# settings #
############

MODEL="Svevo" # Svevo, EuroParl
BLOCK_SIZE=128
LM_EPOCHS=2

################
# fine-tune LM #
################

source ../venv/bin/activate
python3 preprocess_pplm_data.py --labels="unlabeled" --model=$MODEL \
		--max_sentence_length=$BLOCK_SIZE

TESTS="../tests/${MODEL}"
export TRAIN_FILE="${TESTS}/datasets/unlabeled_letters_train_${BLOCK_SIZE}.txt"
export TEST_FILE="${TESTS}/datasets/unlabeled_letters_test_${BLOCK_SIZE}.txt"
OUT_DIR="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
OUT="${OUT_DIR}/lm_fine_tuning_out.txt"

mkdir -p $OUT_DIR
python3 transformers/run_language_modeling.py \
    --output_dir=$OUT_DIR --cache_dir="../tests/cache"\
    --model_type=gpt2 --model_name_or_path="LorenzoDeMattei/GePpeTto" \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval --eval_data_file=$TEST_FILE \
    --block_size=$BLOCK_SIZE --save_steps=1000 --save_total_limit=1 \
    --overwrite_output_dir --epochs=$LM_EPOCHS #> $OUT
    # --no_cuda 

deactivate