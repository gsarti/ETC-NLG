#!/bin/bash

############
# settings #
############

MODEL="Svevo"
BLOCK_SIZE=128
LM_EPOCHS=3
DISCRIM_EPOCHS=30
COND_TEXT="Se potessi"
CLASS_LABELS=("LAVORO" "LIVIA")

############
# run PPLM #
############

source ../venv/bin/activate
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
DATA="../data/trained_models"
TESTS="../tests/${MODEL}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
DISCR_META="discriminator_ep=${DISCRIM_EPOCHS}/generic_classifier_head_meta.json"
DISCR_WEIGHTS="discriminator_ep=${DISCRIM_EPOCHS}/generic_classifier_head.pt"
OUT="${TESTS}/discriminator_ep=${DISCRIM_EPOCHS}/pplm_out.txt"
rm $OUT

for class_label in ${CLASS_LABELS[*]}; do
	python3 transformers/run_pplm.py --class_label $class_label --cond_text="${COND_TEXT}" \
		--discrim "generic" --no_cuda \
		--discrim_meta "${TESTS}/${DISCR_META}" \
		--discrim_weights "${TESTS}/${DISCR_WEIGHTS}" \
	    --length 50 --gamma 1.0 --num_iterations 10 --num_samples 3 \
	    --stepsize 0.02 --window_length 0 --horizon_length 5 --top_k 10 \
	    --kl_scale 0.01 --gm_scale 0.99 --repetition_penalty 1.5 --sample >> $OUT
done