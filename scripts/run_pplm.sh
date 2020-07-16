#!/bin/bash

############
# settings #
############

MODEL="Svevo"
DATASET="gold" # gold, contextual, combined
BLOCK_SIZE=128
LM_EPOCHS=5
DISCRIM_EPOCHS=30
# COND_TEXTS="Se ,Io ,A "
LENGTH=50
SAMPLES=1
ITERS=10

############
# run PPLM # 
############

TESTS="../tests/${MODEL}_${DATASET}"
export DATASET="${TESTS}/${MODEL}_${DATASET}_full.txt"
LM_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
DISCR_META="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/generic_classifier_head_meta.json"
DISCR_WEIGHTS="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/generic_classifier_head.pt"
OUT="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/pplm_out.txt"
SAVEDIR="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/"

if [ "${DATASET}"=="gold" ]; then
	CLASS_LABELS='FAMIGLIA,LIVIA,VIAGGI,SALUTE,LETTERATURA,LAVORO'

if [ "${DATASET}"=="contextual" ]; then
	CLASS_LABELS='decembre|tribel|raffreddore|debole|capanna,
				  notte|mattina|piccolo|sera|olga,
				  notte|piccolo|mattina|olga|sera,
				  raffreddore|fatturare|anonimo|earl|scell,
				  scell|halperson|roncegno|finito|scala,
				  senilità|devotissimo|joyce|amicare|carissimo'

if [ "${DATASET}"=="combined" ]; then
	CLASS_LABELS='cartone|capacità|grossissima|pazzo|schopenhauer,
				  cartone|capacità|grossissima|saggiare|pazzo,
				  fabbricare|domenica|marcare|macchina|caldo,
				  murare|gilda|dimenticato|sabato|arco,
				  senilità|amicare|devotissimo|editore|parigi,
				  titina|vero|olga|bisognare|viaggiare'
fi

source ../venv/bin/activate

python3 transformers/run_pplm.py --class_label "${CLASS_LABELS}" --discrim "generic" \
	--uncond \
	--discrim_meta $DISCR_META --discrim_weights $DISCR_WEIGHTS \
    --length $LENGTH --gamma 1.0 --num_iterations $ITERS --num_samples $SAMPLES \
    --stepsize 0.5 --window_length 0 --horizon_length 5 --top_k 10 \
    --kl_scale 0.01 --gm_scale 0.99 --repetition_penalty 1.5 \
    --sample --savedir $SAVEDIR # > $OUT
    # --no_cuda 
	# --cond_text="${COND_TEXTS}" \