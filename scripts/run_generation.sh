#!/bin/bash

############
# settings #
############

PROMPT="Tanto puro che"
LENGTH=50
BLOCK_SIZE=200
SAVE_DIR="Svevo"

############
# generate #
############

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${SAVE_DIR}"
TRAINED_MODELS="../data/trained_models"
MODEL_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}"
mkdir -p $TESTS

python transformers/run_generation.py \
    --model_type=gpt2 --model_name_or_path=$MODEL_NAME \
    --prompt=$PROMPT --length=$LENGTH --num_return_sequences=3 \
    --repetition_penalty=1.2 --temperature=0.8 --k=5 \
    --no_cuda

deactivate