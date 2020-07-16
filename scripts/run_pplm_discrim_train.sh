#!/bin/bash

############ 
# settings #
############

MODEL="Svevo"
DATASET="gold" # gold, contextual, combined
BLOCK_SIZE=128
LM_EPOCHS=5
DISCRIM_EPOCHS=30

#################
# train discrim #
#################	

TESTS="../tests/${MODEL}_${DATASET}"
export DATASET="${TESTS}/${MODEL}_${DATASET}_full.csv"
LM_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
SAVEDIR="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/"
OUT="${SAVEDIR}/discrim_out.txt"

source ../venv/bin/activate
mkdir -p $SAVEDIR

python3 transformers/run_pplm_discrim_train.py --batch_size=64 --epochs=$DISCRIM_EPOCHS\
	--save_model --dataset="generic" --dataset_fp=$DATASET \
	--pretrained_model=$LM_NAME --log_interval=10 --savedir=$SAVEDIR > $OUT
	# --no_cuda \

deactivate