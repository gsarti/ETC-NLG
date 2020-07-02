#!/bin/bash

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${DATE}"
DATA="../data/trained_models"
mkdir -p $TESTS

############
# settings #
############

BLOCK_SIZE=100
export TRAIN_FILE="../data/letters_train.csv"
MODEL_NAME="${TESTS}/geppetto_svevo_blockSize=${BLOCK_SIZE}"

#################
# train discrim #
#################	

python3 transformers/run_pplm_discrim_train.py --batch_size=128 --epochs=10\
	--save_model --dataset="generic" --dataset_fp=$TRAIN_FILE \
	--pretrained_model=$MODEL_NAME \
	# --no_cuda \

deactivate