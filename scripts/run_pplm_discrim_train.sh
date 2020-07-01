#!/bin/bash

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="../tests/${DATE}"
TRAINED_MODELS="../data/trained_models"
mkdir -p $TESTS


############
# settings #
############

export TRAIN_FILE="../data/letters_train.csv"
MODEL_NAME="${TRAINED_MODELS}/geppetto_svevo_blockSize=100"

#################
# train discrim #
#################	

# transformers is bugged, this script only runs on GPU.
python transformers/run_pplm_discrim_train.py --batch_size=32 \
	--save_model --dataset="generic" --dataset_fp=$TRAIN_FILE \
	--pretrained_model=$MODEL_NAME \
	# --no_cuda \

deactivate