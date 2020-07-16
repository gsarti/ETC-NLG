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
TESTS = "../tests/"


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

def _build_labeled_df(df, dataset):

    if dataset == "gold":

        # renaming cols
        df.columns=['TESTO', 'ID', 'DESTINATARIO', 'LUOGO', 'DATA', 'SALUTO_APERTURA ',
                'FORMULA_APERTURA', 'SALUTO_CHIUSURA', 'FORMULA_CHIUSURA', 'FAMIGLIA',
                'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA_GELOSIA_PER_LIVIA', 'FAM_SOLDI',
                'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
                'LETT_SCRITTURA', 'PAROLE_FAMIGLIA', 'PAROLE_VIAGGI', 'PAROLE_SALUTE',
                'PAROLE_LETTERATURA', 'PAROLE_LAVORO']

        classes = ['FAMIGLIA', 'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA_GELOSIA_PER_LIVIA', 
                   'FAM_SOLDI', 'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
                   'LETT_SCRITTURA']

        new_classes = ['FAMIGLIA', 'LIVIA', 'VIAGGI', 'SALUTE', 'LETTERATURA', 'LAVORO']

        # merge classes
        df.fillna(0., inplace=True)
        df['LIVIA'] = df['FAMIGLIA_AMORE_PER_LIVIA'] + df['FAMIGLIA_GELOSIA_PER_LIVIA']
        df['SALUTE'] = df['SALUTE'] + df['SALUTE_FUMO']
        df['LETTERATURA'] = df['LETTERATURA'] + df['LETT_SCRITTURA']

        labeled_text = [{"label":class_name, "text":row["TESTO"]} 
                         for idx, row in df.iterrows()
                         for class_idx, class_name in enumerate(new_classes)
                         if row[class_name]>=1]

        labeled_df = pd.DataFrame(labeled_text)

        print(labeled_df.groupby("label").agg(['count']))

    elif dataset == "contextual" or dataset == "combined":

        classes = list(np.unique(df[["best_topic"]]))

        labeled_text = [{"label":class_name, "text":row["unpreproc_text"]} 
                         for idx, row in df.iterrows()
                         for class_idx, class_name in enumerate(classes)
                         if row[class_name]>=0.6]

        labeled_df = pd.DataFrame(labeled_text)

        print(labeled_df.groupby("label").agg(['count']))

    else:
        raise NotImplementedError()

    return labeled_df


def _save_df(df, csv, txt, filepath, filename):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if csv:

        df.to_csv(filepath+filename+".csv", encoding='utf-8', 
                  header=False, index=False, sep='\t')
    
    if txt:

        f = open(filepath+filename+".txt", "w")
        for sentence in df["text"].tolist():      
            f.write(sentence+str("\n\n"))
        f.close()


def preprocess_labeled_data(dataset):
    random.seed(0)

    if dataset == "gold":

        df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

    elif dataset == "contextual" or dataset == "combined":

        df = pandas.read_csv(DATA+"topic_annotated_svevo_"+dataset+".tsv", sep="\t")

    else:
        raise NotImplementedError()

    labeled_df = _build_labeled_df(df, dataset)

    # train test split
    msk = np.random.rand(len(labeled_df)) < 0.8
    train = labeled_df[msk]
    test = labeled_df[~msk]

    # cut sentences for discrim training
    cut_full = _cut_sentences(labeled_df, max_sentence_length=1000, return_all_splits=False)

    filepath=TESTS+"Svevo_"+str(dataset)+"/"
    filename="Svevo_"+str(dataset)

    ### txt files for LM fine tuning
    _save_df(train, csv=False, txt=True, filepath=filepath, filename=filename+"_train")
    _save_df(test, csv=False, txt=True, filepath=filepath, filename=filename+"_test")

    ### csv files for PPLM discrim training
    _save_df(cut_full, csv=True, txt=False, filepath=filepath, filename=filename+"_full")


preprocess_labeled_data(dataset="combined")

