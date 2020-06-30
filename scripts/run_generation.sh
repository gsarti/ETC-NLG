#!/bin/bash

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${DATE}"
TRAINED_MODELS="../data/trained_models"
mkdir -p $TESTS

############
# settings #
############

PROMPT="Se io potessi"
LENGTH=30
MODEL_NAME="${TRAINED_MODELS}/geppetto_svevo_blockSize=200"
# MODEL_NAME="LorenzoDeMattei/GePpeTto" 

############
# generate #
############

python transformers/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL_NAME \
    --prompt="${PROMPT}" \
    --length=$LENGTH

deactivate