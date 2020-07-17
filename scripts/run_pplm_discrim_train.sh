#!/bin/bash

############ 
# settings #
############

MODEL="Svevo" # Svevo, EuroParl
LABELS="contextual" # gold, contextual, combined
LM_BLOCK_SIZE=128
LM_EPOCHS=50
DISCRIM_BLOCK_SIZE=1500
DISCRIM_EPOCHS=8

#################
# train discrim #
#################	

source ../venv/bin/activate
python3 preprocess_pplm_data.py --labels=$LABELS --model=$MODEL \
		--max_sentence_length=$DISCRIM_BLOCK_SIZE

TESTS="../tests/${MODEL}"
export DATASET="${TESTS}/datasets/${MODEL}_${LABELS}_${DISCRIM_BLOCK_SIZE}.csv"
LM_NAME="${TESTS}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}"
SAVEDIR="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}_${LABELS}_${DISCRIM_BLOCK_SIZE}/"
OUT="${SAVEDIR}/discrim_out.txt"

if [ "${MODEL}"=="Svevo" ]; then 
	EXAMPLE_SENTENCE="Tanto puro che talvolta dubito veramente che si tratti d'amore perché io altrimenti non potrei consegnarti neppure questa carta."

elif [ "${MODEL}"=="EuroParl" ]; then
	EXAMPLE_SENTENCE=""
fi

mkdir -p $SAVEDIR
python3 transformers/run_pplm_discrim_train.py --batch_size=32 --epochs=$DISCRIM_EPOCHS \
	--example_sentence="${EXAMPLE_SENTENCE}" \
	--save_model --dataset="generic" --dataset_fp=$DATASET \
	--pretrained_model=$LM_NAME --log_interval=10 --savedir=$SAVEDIR > $OUT
	# --no_cuda \

deactivate