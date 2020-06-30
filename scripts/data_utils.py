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


def _build_labeled_df(df):

    classes = ['FAMIGLIA', 'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA, GELOSIA_PER_LIVIA', 
               'FAM_SOLDI', 'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
               'LETT_SCRITTURA']

    # labeled_text = []
    # for idx, row in df.iterrows():

        # label = "$".join([str(class_idx) 
        #                 for class_idx, class_name in enumerate(classes)
        #                 if row[class_name]==1])
        # label = np.nan if label=="" else label
        # labeled_text.append({"label":label, "text":row["TESTO"]})

    labeled_text = [{"label":class_idx, "text":row["TESTO"]} 
                     for idx, row in df.iterrows()
                     for class_idx, class_name in enumerate(classes)
                     if row[class_name]==1]

    labeled_df = pd.DataFrame(labeled_text)
    # labeled_df = labeled_df[labeled_df['label'].notna()]

    print(labeled_df.head())
    return labeled_df


def _save_df(df, csv, txt, tsv, filename):

    if csv:

        df.to_csv(DATA+filename+".csv", encoding='utf-8', index=False, header=False, sep='"', mode='a')
    
    if txt:

        f = open(DATA+filename+".txt", "w")
        for sentence in df["text"].tolist():      
            f.write(sentence+str("\n\n"))
        f.close()

    if tsv:

        with open(DATA+filename+".csv",'r') as csvin:
            with open(DATA+filename+".tsv", 'w') as tsvout:
                csvin = csv_lib.reader(csvin)
                tsvout = csv_lib.writer(tsvout, delimiter='\t')

                for row in csvin:
                    tsvout.writerow(row) 


def preprocess_labeled_data(csv=True, txt=True, tsv=True):
    random.seed(0)

    df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

    # renaming cols
    df.columns=['TESTO', 'ID', 'DESTINATARIO', 'LUOGO', 'DATA', 'SALUTO_APERTURA ',
            'FORMULA_APERTURA', 'SALUTO_CHIUSURA', 'FORMULA_CHIUSURA', 'FAMIGLIA',
            'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA, GELOSIA_PER_LIVIA', 'FAM_SOLDI',
            'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
            'LETT_SCRITTURA', 'PAROLE_FAMIGLIA', 'PAROLE_VIAGGI', 'PAROLE_SALUTE',
            'PAROLE_LETTERATURA', 'PAROLE_LAVORO']

    labeled_df = _build_labeled_df(df)

    # train test split
    msk = np.random.rand(len(labeled_df)) < 0.8
    train = labeled_df[msk]
    test = labeled_df[~msk]

    _save_df(train, csv, txt, tsv, filename="letters_train")
    _save_df(test, csv, txt, tsv, filename="letters_test")
             

preprocess_labeled_data()