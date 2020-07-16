#!/bin/bash
 
############
# settings #
############

MODEL="Svevo"
BLOCK_SIZE=128
EPOCHS=10
LENGTH=50
PROMPT="Tanto puro che"

############
# generate #
############

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${MODEL}"
TRAINED_MODELS="../data/trained_models"
MODEL_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${EPOCHS}"
mkdir -p $TESTS
OUT="${MODEL_NAME}/lm_generation_out.txt"

python transformers/run_generation.py \
    --model_type=gpt2 --model_name_or_path=$MODEL_NAME \
    --prompt="${PROMPT}" --length=$LENGTH --num_return_sequences=3 \
    --repetition_penalty=1.2 --temperature=0.8 --k=5 > $OUT
    # --no_cuda

deactivate