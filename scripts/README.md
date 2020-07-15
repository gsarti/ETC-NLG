# Instructions

```shell
# Preprocess the Svevo corpus (default)
python scripts/preprocess.py

# Preprocess the EuroParl corpora
# Corpus size: 1'909'000
# Vocabulary size: EN = 27249, IT = 31305
python scripts/preprocess.py \
    --corpus_path data/europarl-v7.it-en.en \
    --out_preproc_path data/preprocessed_europarl_en.txt \
    --valid_upos ADJ NOUN VERB PROPN \
    --stopwords "" \
    --no_below 5 \
    --no_above 0.1 \
    --language en \
    --chunksize 1000

python scripts/preprocess.py \
    --corpus_path data/europarl-v7.it-en.it \
    --out_preproc_path data/preprocessed_europarl_it.txt \
    --valid_upos ADJ NOUN VERB PROPN \
    --stopwords "" \
    --no_below 5 \
    --no_above 0.1 \
    --language it \
    --chunksize 1000
```