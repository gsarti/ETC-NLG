import pandas
import os
import numpy as np
import time
import sys
import random
import csv as csv_lib
from sklearn import preprocessing
import pandas as pd
import argparse

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


def _cut_sentences_df(df, labeled, max_sentence_length, return_all_splits):

    if labeled:

        # print("\nUnique labels:\n", np.unique(df[["label"]]))
        cut_sentences = [{"label":row["label"], "text": cut_sentence}
                            for idx, row in df.iterrows()
                            for cut_sentence in _split_string(str=row["text"], 
                            max_sentence_length=max_sentence_length, 
                            return_all_splits=return_all_splits)]

    else:   
        cut_sentences = [{"text": cut_sentence}
                            for idx, row in df.iterrows()
                            for cut_sentence in _split_string(str=row["text"], 
                            max_sentence_length=max_sentence_length, 
                            return_all_splits=return_all_splits)]

    cut_df = pd.DataFrame(cut_sentences)
    return cut_df

def _build_labeled_df(df, labels):

    print("\n== Preprocessing labeled dataset", labels," ==")

    if labels == "gold":

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

    elif labels == "contextual" or labels == "combined":

        classes = list(np.unique(df[["best_topic"]]))

        labeled_text = [{"label":class_name, "text":row["unpreproc_text"]} 
                         for idx, row in df.iterrows()
                         for class_idx, class_name in enumerate(classes)
                         if row[class_name]>=0.6]

    else:
        raise NotImplementedError()

    labeled_df = pd.DataFrame(labeled_text, columns=["label","text"])

    print(labeled_df.groupby("label").agg(['count']))

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


def preprocess_labeled_data(model, max_sentence_length, labels):

    if model=="Svevo":
    
        ### txt files for LM fine tuning

        if labels == "unlabeled":

            print("\n== Preprocessing unlabeled dataset ==")

            df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")[["TESTO"]]
            df.columns = ["text"]

            random.seed(0)
            msk = np.random.rand(len(df)) < 0.8
            train = df[msk]
            test = df[~msk]

            filepath=TESTS+"Svevo/datasets/"
            filename="unlabeled_letters"

            cut_train = _cut_sentences_df(train, labeled=False, 
                max_sentence_length=max_sentence_length, return_all_splits=True)
            cut_test = _cut_sentences_df(test, labeled=False, 
                max_sentence_length=max_sentence_length, return_all_splits=True)

            _save_df(cut_train, csv=False, txt=True, filepath=filepath, 
                filename=filename+"_train_"+str(max_sentence_length))
            _save_df(cut_test, csv=False, txt=True, filepath=filepath,
                filename=filename+"_test_"+str(max_sentence_length))

            print(cut_train.head())

        ### csv files for PPLM discrim training

        else:

            if labels == "gold":

                df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

            elif labels == "contextual" or labels == "combined":

                df = pandas.read_csv(DATA+"topic_annotated_svevo_"+labels+".tsv", sep="\t")

            labeled_df = _build_labeled_df(df, labels)

            # cut sentences for discrim training
            cut_full = _cut_sentences_df(labeled_df, labeled=True, 
                max_sentence_length=max_sentence_length, return_all_splits=False)

            filepath=TESTS+"Svevo/datasets/"
            filename="Svevo_"+str(labels)
            _save_df(cut_full, csv=True, txt=False, filepath=filepath, 
                filename=filename+"_"+str(max_sentence_length))
    
    else:
        raise NotImplementedError()

def main(args):
    preprocess_labeled_data(args.model, args.max_sentence_length, args.labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Svevo")
    parser.add_argument("--max_sentence_length", type=int)
    parser.add_argument("--labels", type=str, default="unlabeled")
    args = parser.parse_args()
    main(args)