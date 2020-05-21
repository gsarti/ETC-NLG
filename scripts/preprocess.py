import spacy
import logging
import pandas as pd
from tqdm import tqdm
from collections import Counter

ORIG_CORPUS_PATH = 'data/carteggio_svevo.csv'
ANNOTATED_CORPUS_PATH = 'data/classificazione_lettere.xlsx'
PREPROC_TEXTS = 'data/preprocessed_texts.txt'
UNPREPROC_TEXTS = 'data/unpreprocessed_texts.txt'

OPEN_UPOS = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']
SVEVO_STOPWORDS = [
    'schmitz', 'signore', 'signora', 'mano', 'ettore', 'lettera', 'parola', 'fare', 'cosa'
]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def preprocess_texts(texts,
    stopwords = [],
    valid_pos = OPEN_UPOS,
    no_below = 5,
    no_above = 0.5):
    """
    Text preprocessing
    """
    nlp = spacy.load("it")
    preprocessed = []
    logger.info('Parsing text with spaCy...')
    for text in tqdm(texts):
        doc = nlp(text.lower())
        preprocessed.append([
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha
            and token.pos_ in valid_pos
            and token.lemma_ not in stopwords
        ])
    logger.info('Filtering extreme occurrences...')
    no_above_abs = int(no_above * len(texts))
    all_words = [w for text in preprocessed for w in text]
    c = Counter(all_words)
    valid_words = [w[0] for w in c.most_common() if no_below <= w[1] <= no_above_abs]
    logger.info(f'Vocabulary size after filtering: {len(valid_words)}')
    preprocessed = [
        " ".join([w for w in text if w in valid_words])
        for text in preprocessed
    ]
    return preprocessed

def main():
    data = pd.read_csv(ORIG_CORPUS_PATH, sep=';', parse_dates=['date'])
    data_it = data[data.mainLanguage == "ITA"]
    with open(UNPREPROC_TEXTS, 'w+') as f:
        for text in data_it['text']:
            f.write(text)
            f.write('\n')
    logger.info(f'Saved unpreprocessed corpus of length '
                f'{len(data_it["text"])} to {UNPREPROC_TEXTS}')
    preproc = preprocess_texts(
        data_it['text'],
        stopwords=SVEVO_STOPWORDS,
        valid_pos=['ADJ', 'NOUN', 'PROPN']
    )
    with open(PREPROC_TEXTS, 'w+') as f:
        for text in preproc:
            f.write(text)
            f.write('\n')
    logger.info(f'Saved unpreprocessed corpus of length '
                f'{len(preproc)} to {PREPROC_TEXTS}')



if __name__ == "__main__":
    main()