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
MODEL_NAME="${TRAINED_MODELS}/geppetto_svevo_blockSize=200"
# PROMPT=""

#################
# train discrim #
#################	

python transformers/run_pplm_discrim_train.py --batch_size=64 \
	--save_model --dataset="generic" --dataset_fp=$TRAIN_FILE \
	--no_cuda \
	# --pretrained_model=$MODEL_NAME \

# python transformers/run_pplm.py -B 1 --cond_text PROMPT \
#     --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 \
#     --kl_scale 0.01 --gm_scale 0.99 --colorama --sample

deactivate