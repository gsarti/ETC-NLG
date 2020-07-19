#!/bin/bash
 
############
# settings #
############

MODEL="EuroParlIta" # Svevo, EuroParlIta, EuroParlEng
LM_BLOCK_SIZE=128
LM_EPOCHS=2
LENGTH=100
PROMPT="Se potessi"

############
# generate #
############

TESTS="../tests/${MODEL}"
MODEL_NAME="${TESTS}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}"
OUT="${MODEL_NAME}/lm_generation_out.txt"

source ../venv/bin/activate
mkdir -p $TESTS

python transformers/run_generation.py \
    --model_type=gpt2 --model_name_or_path=$MODEL_NAME \
    --prompt="${PROMPT}" --length=$LENGTH --num_return_sequences=3 \
    --repetition_penalty=1.2 --temperature=0.8 --k=5 #> $OUT
    # --no_cuda

deactivate