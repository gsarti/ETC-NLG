import pandas
import os
import numpy as np
import time
import sys
import random
import csv as csv_lib
from sklearn import preprocessing
import pandas as pd


DATA = "../data/"
TESTS = "../tests/"+str(time.strftime('%Y-%m-%d'))+"/"
TRAINED_MODELS=DATA+"trained_models/"


def _split_string(str, max_sentence_length, return_all_splits, sep=" "):
    words = str.split()
    if max(map(len, words)) > max_sentence_length:
        raise ValueError("limit is too small")
    res, part, others = [], words[0], words[1:]
    for word in others:
        if len(sep)+len(word) > max_sentence_length-len(part):
            res.append(part)
            part = word
        else:
            part += sep+word
    if part:
        res.append(part)

    return res if return_all_splits else [res[0]]

def _cut_sentences(labeled_df, max_sentence_length, return_all_splits):

    cut_sentences = [{"label":row["label"], "text": cut_sentence}
                        for idx, row in labeled_df.iterrows()
                        for cut_sentence in _split_string(str=row["text"], 
                        max_sentence_length=max_sentence_length, 
                        return_all_splits=return_all_splits)]

    cut_df = pd.DataFrame(cut_sentences)

    print(cut_df.head())
    print("\nUnique labels:\n", np.unique(cut_df[["label"]]))
    return cut_df

def _build_labeled_df(df):

    classes = ['FAMIGLIA', 'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA_GELOSIA_PER_LIVIA', 
               'FAM_SOLDI', 'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
               'LETT_SCRITTURA']

    labeled_text = [{"label":class_name, "text":row["TESTO"]} 
                     for idx, row in df.iterrows()
                     for class_idx, class_name in enumerate(classes)
                     if row[class_name]==1]

    labeled_df = pd.DataFrame(labeled_text)
    # labeled_df = labeled_df[labeled_df['label'].notna()]

    return labeled_df

def _save_df(df, csv, txt, filename):

    if csv:

        df.to_csv(DATA+filename+".csv", encoding='utf-8', 
                  header=False, index=False, sep='\t')
    
    if txt:

        f = open(DATA+filename+".txt", "w")
        for sentence in df["text"].tolist():      
            f.write(sentence+str("\n\n"))
        f.close()


def preprocess_labeled_data():
    random.seed(0)

    df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

    # renaming cols
    df.columns=['TESTO', 'ID', 'DESTINATARIO', 'LUOGO', 'DATA', 'SALUTO_APERTURA ',
            'FORMULA_APERTURA', 'SALUTO_CHIUSURA', 'FORMULA_CHIUSURA', 'FAMIGLIA',
            'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA_GELOSIA_PER_LIVIA', 'FAM_SOLDI',
            'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
            'LETT_SCRITTURA', 'PAROLE_FAMIGLIA', 'PAROLE_VIAGGI', 'PAROLE_SALUTE',
            'PAROLE_LETTERATURA', 'PAROLE_LAVORO']

    labeled_df = _build_labeled_df(df)

    # train test split
    msk = np.random.rand(len(labeled_df)) < 0.8
    train = labeled_df[msk]
    test = labeled_df[~msk]

    ### txt files for LM fine tuning
    _save_df(train, csv=False, txt=False, filename="letters_train")
    _save_df(test, csv=False, txt=False, filename="letters_test")

    ### csv files for PPLM training
    cut_train = _cut_sentences(train, max_sentence_length=1000, return_all_splits=False)
    cut_test = _cut_sentences(test, max_sentence_length=1000, return_all_splits=False)
    _save_df(cut_train, csv=True, txt=True, filename="letters_train")
    _save_df(cut_test, csv=True, txt=True, filename="letters_test")


preprocess_labeled_data()