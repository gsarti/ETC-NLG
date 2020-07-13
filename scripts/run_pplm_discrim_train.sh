#!/bin/bash

############
# settings #
############

export TRAIN_FILE="../data/letters_train.csv"
BLOCK_SIZE=256
SAVE_DIR="Svevo"

#################
# train discrim #
#################	

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
DATA="../data/trained_models"
TESTS="../tests/${SAVE_DIR}"
SAVEDIR="${TESTS}/discriminator/"
mkdir -p $SAVEDIR
MODEL_NAME="${TESTS}/fine_tuned_LM_blockSize=${BLOCK_SIZE}"

python3 transformers/run_pplm_discrim_train.py --batch_size=64 --epochs=20\
	--save_model --dataset="generic" --dataset_fp=$TRAIN_FILE \
	--pretrained_model=$MODEL_NAME --log_interval=10 --savedir=$SAVEDIR\
	# --no_cuda \

deactivate