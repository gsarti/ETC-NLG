#!/bin/bash

############
# settings #
############

SAVE_DIR="Svevo"
DISCR_META="discriminator/generic_classifier_head_meta.json"
DISCR_WEIGHTS="discriminator/generic_classifier_head.pt"

############
# run PPLM #
############

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
DATA="../data/trained_models"
DISCRIM="${DATA}"
TESTS="../tests/${SAVE_DIR}"
mkdir -p $TESTS

python3 transformers/run_pplm.py --class_label 5 --cond_text="Se potessi" \
	--discrim "generic" --no_cuda \
	--discrim_meta "${TESTS}/${DISCR_META}" \
	--discrim_weights "${TESTS}/${DISCR_WEIGHTS}" \
    --length 50 --gamma 1.0 --num_iterations 10 --num_samples 1 \
    --stepsize 0.03 --window_length 10 --horizon_length 10 --top_k 10 \
    --kl_scale 0.01 --gm_scale 0.99 --repetition_penalty 1.5 --sample

deactivate