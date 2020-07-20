# Instructions

## Topic Models

```shell
# Preprocess the Svevo corpus (default)
# Corpus size: 826 paragraphs
# Vocabulary size: 2014
python scripts/preprocess.py

# Preprocess the EuroParl corpora
# We consider only the top 50'000 sentences
# Corpus size: 1'909'115 sentences
# Vocabulary size: EN = 27'250, IT = 31'305
python scripts/preprocess.py \
    --corpus_path data/europarl-v7.it-en.en \
    --out_preproc_path data/preprocessed_europarl_en.txt \
    --out_unpreproc_path data/unpreprocessed_europarl_en.txt \
    --valid_upos ADJ NOUN VERB PROPN \
    --stopwords "" \
    --no_below 5 \
    --no_above 0.1 \
    --language en \
    --chunksize 1000 \
    --topk 50000

python scripts/preprocess.py \
    --corpus_path data/europarl-v7.it-en.it \
    --out_preproc_path data/preprocessed_europarl_it.txt \
    --out_unpreproc_path data/unpreprocessed_europarl_it.txt \
    --valid_upos ADJ NOUN VERB PROPN \
    --stopwords "" \
    --no_below 5 \
    --no_above 0.1 \
    --language it \
    --chunksize 1000 \
    --topk 50000

# Generates and evaluates topic models in range 3-10 topics for
# the Svevo corpus
python scripts/eval_topic_models.py

# Do it only for topic size 6 using both models
python scripts/eval_topic_models.py \
    --n_topics 6 \
    --modes contextual combined

# Eval topic models on the Europarl corpora
python scripts/eval_topic_models.py \
    --preproc_path data/preprocessed_europarl_en.txt \
    --unpreproc_path data/unpreprocessed_europarl_en.txt \
    --embed_model_name roberta-base \
    --model_identifier europarl_en \
    --language en \
    --n_topics 25 50 75 100 150

python scripts/eval_topic_models.py \
    --preproc_path data/preprocessed_europarl_it.txt \
    --unpreproc_path data/unpreprocessed_europarl_it.txt \
    --model_identifier europarl_it \
    --language it \
    --n_topics 25 50 75 100 150

# Create the topic-labeled Svevo dataset for the contextual model
python scripts/label_unpreprocessed.py

# Create the topic-labeled Svevo dataset for the combined model
python scripts/label_unpreprocessed.py \
    --inference_type combined \
    --save_path data/topic_annotated_svevo_combined.tsv

# Create topic-labeled Svevo dataset for contextual and combined models
# for Europarl IT and EN using the best scoring topic model.
python scripts/label_unpreprocessed.py \
    --preproc_path data/preprocessed_europarl_en.txt \
    --unpreproc_path data/unpreprocessed_europarl_en.txt \
    --embeds_path models/preprocessed_europarl_en_roberta-base \
    --model_dir models/ctm_europarl_en_75_100_contextual \
    --save_path data/topic_annotated_europarl_en_contextual.tsv \
    --inference_type contextual

python scripts/label_unpreprocessed.py \
    --preproc_path data/preprocessed_europarl_it.txt \
    --unpreproc_path data/unpreprocessed_europarl_it.txt \
    --embeds_path models/preprocessed_europarl_it_umberto-commoncrawl-cased-v1 \
    --model_dir models/ctm_europarl_it_75_100_contextual \
    --save_path data/topic_annotated_europarl_it_contextual.tsv \
    --inference_type contextual

```

## Plug and Play Models


`run_lm_fine_tuning.sh` performs fine-tuning on a pre-trained language model:
*GePpeTto* for `Svevo` and `EuroParlIta` datasets and *GPT-2* for `EuroParlEng` datasets.

`run_generation.sh` generates sentences from a fine-tuned language model.

`run_pplm_discrim_train.sh` trains a discriminator on a labeled dataset.

`run_pplm.sh` used a discriminator model to generate text conditioned on an input prefix string and a conditioning class.


### Scripts settings 

- `MODEL` refers to the base dataset for language modelling. It can be either "Svevo", "EuroParlIta" or "EuroParlEng"
- `LM_BLOCK_SIZE` is the maximum allowed sequence length after tokenization during fine-tuning
- `LM_EPOCHS` is the number of epochs for fine-tuning
- `LENGTH` is the lenght of the generated string
- `PROMPT` is the input prefix for the generation of sentences
- `LABELS` is the chosen class of labels, which can be either:
    - "gold" is a human-labeled dataset
    - "contextual" is labeled by a *contextual topic model*
    - "combined" is labeled by a *combined topic model*
- `DISCRIM_BLOCK_SIZE` is the maximum lenght of sequences for the discriminator
- `DISCRIM_EPOCHS` is the number of training epochs for the discriminator
- `SAMPLES` is the number of sentences produced by a Plug and Play model for each conditioning class and conditioning input
- `ITERS` is the number of iterations of Plug and Play model through the conditioned text

**Additional options**
- pass `--no_cuda` flag to execute on CPU
- pass `--uncond` instead of `--cond_text` to `run_pplm.py` for generating strings without prefix 


### Scripts outputs

All the trained models, output texts and datasets are saved in `../tests/` directory, with the following structure:

- tests/
    - Svevo/
        - datasets/
        - fine_tuned_LM/
            - discriminator_gold
            - ...
    - EuroParlIta/
        - ...    
    - EuroParlEng/
        - ...