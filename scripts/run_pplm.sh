#!/bin/bash

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${DATE}"
DATA="../data/trained_models"
DISCRIM="${DATA}"
mkdir -p $TESTS

############
# settings #
############

DISCR_META="${TESTS}/discriminator/generic_classifier_head_meta.json"
DISCR_WEIGHTS="${TESTS}/discriminator/generic_classifier_head.pt"

############
# run PPLM #
############

python3 transformers/run_pplm.py --class_label 8 --cond_text="Tanto puro che talvolta" \
	--discrim "generic" --no_cuda \
	--discrim_meta $DISCR_META --discrim_weights $DISCR_WEIGHTS\
    --length 100 --gamma 1.5 --num_iterations 10 --num_samples 10 --stepsize 0.03 --window_length 5 \
    --kl_scale 0.01 --gm_scale 0.99 --sample

deactivate