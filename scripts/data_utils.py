import pandas
import os
import numpy as np
import time
import sys
import random

DATA = "../data/"
TESTS = "../tests/"+str(time.strftime('%Y-%m-%d'))+"/"
TRAINED_MODELS=DATA+"trained_models/"
    
def preprocess_labeled_data():
    random.seed(0)

    df = pandas.read_excel(DATA+"classificazione_lettere.xlsx")

    df.columns=['TESTO', 'ID', 'DESTINATARIO', 'LUOGO', 'DATA', 'SALUTO_APERTURA ',
            'FORMULA_APERTURA', 'SALUTO_CHIUSURA', 'FORMULA_CHIUSURA', 'FAMIGLIA',
            'FAMIGLIA_AMORE_PER_LIVIA', 'FAMIGLIA, GELOSIA_PER_LIVIA', 'FAM_SOLDI',
            'VIAGGI', 'SALUTE', 'SALUTE_FUMO', 'LETTERATURA', 'LAVORO',
            'LETT_SCRITTURA', 'PAROLE_FAMIGLIA', 'PAROLE_VIAGGI', 'PAROLE_SALUTE',
            'PAROLE_LETTERATURA', 'PAROLE_LAVORO']
    df.to_excel(DATA+"labeled_letters.xlsx", encoding='utf-8', index=False)
    
    text = df[["TESTO"]]
    msk = np.random.rand(len(text)) < 0.8
    train = text[msk]
    test = text[~msk]

    train.to_csv(DATA+"letters_train.csv", encoding='utf-8', index=False, header=False, sep='"', mode='a')
    test.to_csv(DATA+"letters_test.csv", encoding='utf-8', index=False, header=False, sep='"', mode='a')
    
    f = open(DATA+"letters_train.txt", "w")
    for sentence in train["TESTO"].tolist():      
        f.write(str("[START]")+sentence+str("[STOP]\n\n"))
    f.close()

    f = open(DATA+"letters_test.txt", "w")
    for sentence in test["TESTO"].tolist():      
        f.write(str("[START]")+sentence+str("[STOP]\n\n"))
    f.close()


preprocess_labeled_data()