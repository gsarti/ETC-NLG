```shell
python scripts/preprocess.py \
    --corpus_path data/europarl-v7.it-en.en \
    --out_preproc_path data/preprocessed_europarl_en.txt \
    --valid_upos ADJ NOUN VERB PROPN \
    --stopwords "" \
    --no_below 5 \
    --no_above 0.1 \
    --language en
```