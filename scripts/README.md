# Instructions

```shell
# Preprocess the Svevo corpus (default)
# Corpus size: 826 paragraphs
# Vocabulary size: 2014
python scripts/preprocess.py

# Preprocess the EuroParl corpora
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
```
