#!/bin/bash

############
# settings #
############

export TRAIN_FILE="../data/letters_train.csv"
MODEL="Svevo"
BLOCK_SIZE=512
LM_EPOCHS=5
DISCRIM_EPOCHS=50

#################
# train discrim #
#################	

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
DATA="../data/trained_models"
LM_NAME="../tests/${MODEL}/fine_tuned_LM_blockSize=${BLOCK_SIZE}_ep=${LM_EPOCHS}"
SAVEDIR="${LM_NAME}/discriminator_ep=${DISCRIM_EPOCHS}/"
OUT="${SAVEDIR}/discrim_out.txt"
mkdir -p $SAVEDIR

python3 transformers/run_pplm_discrim_train.py --batch_size=64 --epochs=$DISCRIM_EPOCHS\
	--save_model --dataset="generic" --dataset_fp=$TRAIN_FILE \
	--pretrained_model=$LM_NAME --log_interval=10 --savedir=$SAVEDIR > $OUT
	# --no_cuda \

deactivate