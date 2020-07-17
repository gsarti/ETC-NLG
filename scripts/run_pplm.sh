#!/bin/bash

############
# settings #
############

MODEL="Svevo" # Svevo, EuroParl
LABELS="contextual" # gold, contextual, combined
LM_BLOCK_SIZE=128
LM_EPOCHS=50
DISCRIM_BLOCK_SIZE=1500
DISCRIM_EPOCHS=20
LENGTH=50
SAMPLES=3
ITERS=10

############
# run PPLM # 
############

DISCR_PATH="../tests/${MODEL}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}/discriminator_ep=${DISCRIM_EPOCHS}_${LABELS}_${DISCRIM_BLOCK_SIZE}"
DISCR_META="${DISCR_PATH}/generic_classifier_head_meta.json"
DISCR_WEIGHTS="${DISCR_PATH}/generic_classifier_head.pt"
OUT="${DISCR_PATH}/pplm_out.txt"
SAVEDIR="${DISCR_PATH}/"

if [ "${MODEL}"=="Svevo" ]; then

	COND_TEXTS="Se potessi,Io sono,La tua,Un giorno"

	if [ "${DATASET}"=="gold" ]; then
		CLASS_LABELS='FAMIGLIA,LIVIA,VIAGGI,SALUTE,LETTERATURA,LAVORO'

	elif [ "${DATASET}"=="contextual" ]; then
		CLASS_LABELS='decembre|tribel|raffreddore|debole|capanna,
					  notte|mattina|piccolo|sera|olga,
					  notte|piccolo|mattina|olga|sera,
					  raffreddore|fatturare|anonimo|earl|scell,
					  scell|halperson|roncegno|finito|scala,
					  senilità|devotissimo|joyce|amicare|carissimo'

	elif [ "${DATASET}"=="combined" ]; then
		CLASS_LABELS='cartone|capacità|grossissima|pazzo|schopenhauer,
					  cartone|capacità|grossissima|saggiare|pazzo,
					  fabbricare|domenica|marcare|macchina|caldo,
					  murare|gilda|dimenticato|sabato|arco,
					  senilità|amicare|devotissimo|editore|parigi,
					  titina|vero|olga|bisognare|viaggiare'
	fi

elif [ "${MODEL}"=="EuroParl" ]; then

	COND_TEXTS=""

	if [ "${DATASET}"=="gold" ]; then
		CLASS_LABELS=""

	elif [ "${DATASET}"=="contextual" ]; then
		CLASS_LABELS=""

	elif [ "${DATASET}"=="contextual" ]; then
		CLASS_LABELS=""

	fi

fi

source ../venv/bin/activate

python3 transformers/run_pplm.py --class_label "${CLASS_LABELS}" --discrim "generic" \
	--cond_text="${COND_TEXTS}" \
	--discrim_meta $DISCR_META --discrim_weights $DISCR_WEIGHTS \
    --length $LENGTH --gamma 1.0 --num_iterations $ITERS --num_samples $SAMPLES \
    --stepsize 0.05 --window_length 0 --horizon_length 5 --top_k 10 \
    --kl_scale 0.01 --gm_scale 0.99 --repetition_penalty 1.5 \
    --sample --savedir $SAVEDIR #> $OUT

	# --uncond \
    # --no_cuda 
