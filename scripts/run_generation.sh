#!/bin/bash
 
############
# settings #
############

MODEL="Svevo" # Svevo, EuroParl
BLOCK_SIZE=128
EPOCHS=2
LENGTH=50
PROMPT="Se potessi"

############
# generate #
############

TESTS="../tests/${MODEL}"
MODEL_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${EPOCHS}"
OUT="${MODEL_NAME}/lm_generation_out.txt"

source ../venv/bin/activate
mkdir -p $TESTS

python transformers/run_generation.py \
    --model_type=gpt2 --model_name_or_path=$MODEL_NAME \
    --prompt="${PROMPT}" --length=$LENGTH --num_return_sequences=3 \
    --repetition_penalty=1.2 --temperature=0.8 --k=5 # > $OUT
    # --no_cuda

deactivate