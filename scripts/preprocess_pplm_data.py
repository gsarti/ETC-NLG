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
import re

DATA = "../data/"
TESTS = "../tests/"

def _split_with_punctuation(string, max_sentence_length, return_all_splits, sep=" "):

    def split_keep(string, sep):
        return re.findall('[^'+sep+']+'+sep+'|[^'+sep+']+', string)
    
    if max_sentence_length < 8:
            raise ValueError("limit is too small")

    if split_keep(string, "."):
        split_string = []
        for sentence in split_keep(string, "."):
            if sentence:
                for subsentence in split_keep(sentence,","):
                    if subsentence:
                        split_string.append(subsentence)
                    else:
                        split_string.append(sentence)
    else:
        split_string = string.split()

    res = []
    part = split_string[0]
    others = split_string[1:]
    sep=" "
    for word in others:
        if len(sep)+len(word) > max_sentence_length-len(part):
            res.append(part)
            part = word
        else:
            part += sep+word
    if part:
        res.append(part)

    return res if return_all_splits else [res[0]]

# def _split_string(string, max_sentence_length, return_all_splits, sep=" "):
#     words = string.split()

#     if max_sentence_length < 8:
#         raise ValueError("limit is too small")
#     res, part, others = [], words[0], words[1:]
#     for word in others:
#         if len(sep)+len(word) > max_sentence_length-len(part):
#             res.append(part)
#             part = word
#         else:
#             part += sep+word
#     if part:
#         res.append(part)

#     return res if return_all_splits else [res[0]]


def _cut_sentences_df(df, labeled, max_sentence_length, return_all_splits):

    if labeled:

        cut_sentences = [{"label":row["label"], "text": cut_sentence}
                            for idx, row in df.iterrows()
                            for cut_sentence in _split_with_punctuation(string=row["text"], 
                            max_sentence_length=max_sentence_length, 
                            return_all_splits=return_all_splits)]

    else:   
        cut_sentences = [{"text": cut_sentence}
                            for idx, row in df.iterrows()
                            for cut_sentence in _split_with_punctuation(string=row["text"], 
                            max_sentence_length=max_sentence_length, 
                            return_all_splits=return_all_splits)
                           ]

    cut_df = pd.DataFrame(cut_sentences)
    return cut_df

def _build_labeled_df(df, model, labels):


    if labels == "gold":

        if model == "Svevo":

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

        elif model == "EuroParlIta" or model == "EuroParlEng":

            raise NotImplementedError()

    elif labels == "contextual" or labels == "combined":

        unique_topics = np.unique(df[["best_topic"]])
        classes = list(unique_topics)


        labeled_text = [{"label":class_name, "text":row["unpreproc_text"]} 
                         for idx, row in df.iterrows()
                         for class_idx, class_name in enumerate(classes)
                         if row[class_name]>=0.6]
    
    else:
        raise NotImplementedError()

    labeled_df = pd.DataFrame(labeled_text, columns=["label","text"])

    # remove rows with empty strings
    # labeled_df['text'].replace('', np.nan, inplace=True)
    # labeled_df.dropna(subset=['text'], inplace=True)

    # select top k labels counts
    labels_counts = labeled_df["label"].value_counts().nlargest(10)
    top_labels = list(labels_counts.index)
    top_labels_df = labeled_df[labeled_df['label'].isin(top_labels)]
    print("\nTop labels counts:\n\n",labels_counts)

    return top_labels_df


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


def load_data(model, max_sentence_length, labels):

    if model=="Svevo":
    
        if labels == "unlabeled":

            df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")[["TESTO"]]
            df.columns = ["text"]

        elif labels == "gold":

            df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

        elif labels == "contextual" or labels == "combined":

            df = pandas.read_csv(DATA+"topic_annotated_svevo_"+labels+"_6.tsv", sep="\t")

    elif model=="EuroParlIta":

        if labels == "unlabeled":
            
            europarl = open('../data/europarl-v7.it-en.it', encoding='utf-8')\
                            .read().split('\n')
            europarl = list(filter(None, europarl))
            europarl = [sentence for sentence in europarl if len(sentence)>8]
            df = pd.DataFrame(europarl, columns=["text"])

        elif labels == "contextual" or labels == "combined":

            filename = "topic_annotated_europarl_it_"+labels+"_10.tsv"
            df = pandas.read_csv(DATA+filename, sep="\t")
            df = df[df['unpreproc_text'].notna()]

    elif model=="EuroParlEng":

        if labels == "unlabeled":
            
            europarl = open('../data/europarl-v7.it-en.en', encoding='utf-8')\
                            .read().split('\n')
            europarl = list(filter(None, europarl))
            europarl = [sentence for sentence in europarl if len(sentence)>8]
            df = pd.DataFrame(europarl, columns=["text"])

        elif labels == "contextual" or labels == "combined":

            filename = "topic_annotated_europarl_en_"+labels+"_10.tsv"
            df = pandas.read_csv(DATA+filename, sep="\t")
            df = df[df['unpreproc_text'].notna()]

    return df


def preprocess_data(df, model, max_sentence_length, labels):

    if labels == "unlabeled": ### txt files for LM fine tuning

        print("\n== Preprocessing unlabeled dataset ==")

        random.seed(0)
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]

        filepath, filename = (TESTS+model+"/datasets/","unlabeled_"+model)

        cut_train = _cut_sentences_df(train, labeled=False, 
            max_sentence_length=max_sentence_length, return_all_splits=True)
        cut_test = _cut_sentences_df(test, labeled=False, 
            max_sentence_length=max_sentence_length, return_all_splits=True)

        _save_df(cut_train, csv=False, txt=True, filepath=filepath, 
            filename=filename+"_train_"+str(max_sentence_length))
        _save_df(cut_test, csv=False, txt=True, filepath=filepath,
            filename=filename+"_test_"+str(max_sentence_length))

        print(cut_train.head())

    else: ### csv files for PPLM discrim training

        print("\n== Preprocessing labeled dataset", labels," ==")

        labeled_df = _build_labeled_df(df, model, labels)

        # cut sentences for discrim training
        cut_full = _cut_sentences_df(labeled_df, labeled=True, 
            max_sentence_length=max_sentence_length, return_all_splits=False)

        filepath, filename = (TESTS+model+"/datasets/", model+"_"+str(labels))
        
        _save_df(cut_full, csv=True, txt=False, filepath=filepath, 
            filename=filename+"_"+str(max_sentence_length))

def main(args):

    df = load_data(args.model, args.max_sentence_length, args.labels)
    preprocess_data(df, args.model, args.max_sentence_length, args.labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Svevo")
    parser.add_argument("--max_sentence_length", type=int, default=128)
    parser.add_argument("--labels", type=str, default="unlabeled")
    args = parser.parse_args()
    main(args)