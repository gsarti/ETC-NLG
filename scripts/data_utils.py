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


def _build_labeled_df(df, limit):

    classes = ['FAMIGLIA', 'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA, GELOSIA_PER_LIVIA', 
               'FAM_SOLDI', 'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
               'LETT_SCRITTURA']

    labeled_text = [{"label":class_idx, "text":row["TESTO"]} 
                     for idx, row in df.iterrows()
                     for class_idx, class_name in enumerate(classes)
                     if row[class_name]==1]

    labeled_df = pd.DataFrame(labeled_text)

    splitted_sentences = [{"label":row["label"], "text": cut_sentence}
                            for idx, row in labeled_df.iterrows()
                            if len(row["text"]) > limit
                            for cut_sentence in split_string(row["text"], limit=limit)]

    splitted_df = pd.DataFrame(splitted_sentences)
    # splitted_df = splitted_df[splitted_df['label'].notna()]

    print(splitted_df.head())
    return splitted_df


def _save_df(df, csv, txt, filename):

    if csv:

        df.to_csv(DATA+filename+".csv", encoding='utf-8', 
                  header=False, index=False, sep='\t')
    
    if txt:

        f = open(DATA+filename+".txt", "w")
        for sentence in df["text"].tolist():      
            f.write(sentence+str("\n\n"))
        f.close()


def preprocess_labeled_data(csv=True, txt=False, limit=100):
    random.seed(0)

    df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

    # renaming cols
    df.columns=['TESTO', 'ID', 'DESTINATARIO', 'LUOGO', 'DATA', 'SALUTO_APERTURA ',
            'FORMULA_APERTURA', 'SALUTO_CHIUSURA', 'FORMULA_CHIUSURA', 'FAMIGLIA',
            'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA, GELOSIA_PER_LIVIA', 'FAM_SOLDI',
            'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
            'LETT_SCRITTURA', 'PAROLE_FAMIGLIA', 'PAROLE_VIAGGI', 'PAROLE_SALUTE',
            'PAROLE_LETTERATURA', 'PAROLE_LAVORO']

    labeled_df = _build_labeled_df(df, limit)

    # train test split
    msk = np.random.rand(len(labeled_df)) < 0.8
    train = labeled_df[msk]
    test = labeled_df[~msk]

    _save_df(train, csv, txt, filename="letters_train")
    _save_df(test, csv, txt, filename="letters_test")


def split_string(str, limit, sep=" "):
    words = str.split()
    if max(map(len, words)) > limit:
        raise ValueError("limit is too small")
    res, part, others = [], words[0], words[1:]
    for word in others:
        if len(sep)+len(word) > limit-len(part):
            res.append(part)
            part = word
        else:
            part += sep+word
    if part:
        res.append(part)
    return res


preprocess_labeled_data(csv=True, limit=300)