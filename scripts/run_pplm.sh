#!/bin/bash

############
# settings #
############

MODEL="EuroParlEng" # Svevo, EuroParlIta, EuroParlEng
LABELS="contextual" # gold (not available on Europarl), contextual, combined
LM_BLOCK_SIZE=128
LM_EPOCHS=2
DISCRIM_BLOCK_SIZE=1500
DISCRIM_EPOCHS=10
LENGTH=150
SAMPLES=1
ITERS=10
TEMP=2.0
GM_SCALE=0.95

############
# run PPLM # 
############

source ../venv/bin/activate

TESTS="../tests/${MODEL}"
DATASET="${TESTS}/datasets/${MODEL}_${LABELS}_${DISCRIM_BLOCK_SIZE}.csv"

if [ ! -f "${DATASET}" ]; then
	python3 preprocess_pplm_data.py --labels=$LABELS --model=$MODEL \
			--max_sentence_length=$DISCRIM_BLOCK_SIZE
fi

export DATASET="${DATASET}"

LM_NAME="${TESTS}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}"
DISCR_PATH="../tests/${MODEL}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}/discriminator_ep=${DISCRIM_EPOCHS}_${LABELS}_${DISCRIM_BLOCK_SIZE}"
DISCR_META="${DISCR_PATH}/generic_classifier_head_meta.json"
DISCR_WEIGHTS="${DISCR_PATH}/generic_classifier_head.pt"
OUT="${DISCR_PATH}/pplm_out_${MODEL}_samp=${SAMPLES}_iters=${ITERS}_temp=${TEMP}_gm=${GM_SCALE}.txt"
SAVEDIR="${DISCR_PATH}/"

if [ "${MODEL}" == "Svevo" ] ; then

	COND_TEXTS="Se potessi,Io sono,La tua,Un giorno"

elif [ "${MODEL}" == "EuroParlIta" ]; then

	COND_TEXTS="Dato il,Si dovrebbe,Penso che,In questo"

elif [ "${MODEL}" == "EuroParlEng" ]; then

	COND_TEXTS="It is,I would" #,You did,In this"

fi


source ../venv/bin/activate

python3 transformers/run_pplm.py --dataset=$DATASET --labels="${LABELS}" --discrim "generic" \
	--pretrained_model=$LM_NAME \
	--cond_text="${COND_TEXTS}" --model="${MODEL}" --temperature=$TEMP \
	--discrim_meta $DISCR_META --discrim_weights $DISCR_WEIGHTS \
    --length $LENGTH --gamma 1.0 --num_iterations $ITERS --num_samples $SAMPLES \
    --stepsize 0.05 --window_length 0 --horizon_length 5 --top_k 10 \
    --kl_scale 0.01 --gm_scale $GM_SCALE --repetition_penalty 1.5 \
    --sample --savedir $SAVEDIR > $OUT

	# --uncond \
    # --no_cuda 
