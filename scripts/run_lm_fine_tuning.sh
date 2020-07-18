#!/bin/bash
 
############
# settings #
############

MODEL="Svevo" # Svevo, EuroParlIta, EuroParlEng
LM_BLOCK_SIZE=128
LM_EPOCHS=2

################
# fine-tune LM #
################

source ../venv/bin/activate

TESTS="../tests/${MODEL}"
TRAIN_FILE="${TESTS}/datasets/unlabeled_${MODEL}_train_${LM_BLOCK_SIZE}.txt"
TEST_FILE="${TESTS}/datasets/unlabeled_${MODEL}_test_${LM_BLOCK_SIZE}.txt"

if [ ! -f "${TRAIN_FILE}" ]; then
    python3 preprocess_pplm_data.py --labels="unlabeled" --model=$MODEL \
    		--max_sentence_length=$LM_BLOCK_SIZE
fi


if [ "${MODEL}"=="Svevo" ] ; then

	BASE_LM_NAME="LorenzoDeMattei/GePpeTto"

elif [ "${MODEL}"=="EuroParlIta" ]; then

	BASE_LM_NAME="LorenzoDeMattei/GePpeTto"

elif [ "${MODEL}"=="EuroParlEng" ]; then

	BASE_LM_NAME="gpt2-medium"

fi

export TRAIN_FILE="${TRAIN_FILE}"
export TEST_FILE="${TEST_FILE}"
OUT_DIR="${TESTS}/fine_tuned_LM_blockSize=${LM_BLOCK_SIZE}_ep=${LM_EPOCHS}"
OUT="${OUT_DIR}/lm_fine_tuning_out.txt"

mkdir -p $OUT_DIR
python3 transformers/run_language_modeling.py \
    --output_dir=$OUT_DIR --cache_dir="../tests/cache"\
    --model_type=gpt2 --model_name_or_path="${BASE_LM_NAME}" \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval --eval_data_file=$TEST_FILE \
    --block_size=$LM_BLOCK_SIZE --save_steps=1000 --save_total_limit=1 \
    --overwrite_output_dir --epochs=$LM_EPOCHS > $OUT
    # --no_cuda 

deactivate