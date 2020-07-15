import spacy
import math
import logging
import argparse
import pandas as pd
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from collections import Counter

ORIG_CORPUS_PATH = 'data/carteggio_svevo.csv'
PREPROC_TEXTS = 'data/preprocessed_svevo_texts.txt'
UNPREPROC_TEXTS = 'data/unpreprocessed_svevo_texts.txt'

VALID_UPOS = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']
SVEVO_STOPWORDS = [
    'schmitz', 'signore', 'signora', 'mano', 'ettore', 'lettera', 'parola', 'fare', 'cosa'
]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length + chunksize - 1, chunksize))


def preprocess_parallel(args, texts):
    args.nlp = spacy.load(args.language)
    executor = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', prefer="processes")
    logger.info('Parsing text with spaCy...')
    do = delayed(preprocess_texts)
    chunks = chunker(texts, len(texts), chunksize=args.chunksize)
    max_range = int(math.ceil(len(texts) / args.chunksize))
    tasks = (do(args, chunk) for i, chunk in zip(trange(max_range),chunks))
    result = executor(tasks)
    result = [sentence for sublist in result for sentence in sublist]
    logger.info('Filtering extreme occurrences...')
    no_above_abs = int(args.no_above * len(texts))
    all_words = [w for text in result for w in text]
    c = Counter(all_words)
    valid_words = [w[0] for w in c.most_common() if args.no_below <= w[1] <= no_above_abs]
    logger.info(f'Vocabulary size after filtering: {len(valid_words)}')
    logger.info("Joining valid words per document...")
    preprocessed = [
        " ".join([w for w in text if w in valid_words])
        for text in tqdm(result)
    ]
    return preprocessed


def preprocess_texts(args, texts):
    return [[token.lemma_ for token in args.nlp(text.lower())
                if not token.is_stop and token.is_alpha
                and token.pos_ in args.valid_upos
                and token.lemma_ not in args.stopwords ] for text in texts]


def main(args):
    # We want to create the unpreprocessed corpus for Svevo epistolary
    # 1 sentence per row
    logger.info(f"Script arguments: {args}")
    if args.corpus_path == ORIG_CORPUS_PATH:
        data = pd.read_csv(args.corpus_path, sep=';', parse_dates=['date'])
        data = data[data.mainLanguage == "ITA"]
        with open(args.out_unpreproc_path, 'w+') as f:
            for text in data['text']:
                f.write(text)
                f.write('\n')
        logger.info(f'Saved unpreprocessed corpus of length '
                    f'{len(data["text"])} to {args.out_unpreproc_path}')
        sentences = data["text"]
    # We consider the corpus to be already in the right shape otherwise
    else:
        with open(args.corpus_path, 'r') as f:
            sentences = f.read().splitlines()
    logger.info(f'Corpus length before preprocessing: {len(sentences)}')
    preproc = preprocess_parallel(args, sentences)
    with open(args.out_preproc_path, 'w+') as f:
        for text in preproc:
            f.write(text)
            f.write('\n')
    logger.info(f'Saved preprocessed corpus of length '
                f'{len(preproc)} to {args.out_preproc_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        default=ORIG_CORPUS_PATH,
        type=str,
        help="The path to the input corpus. By default, use the Svevo epistolary corpus: %(default)s.",
    )
    parser.add_argument(
        "--out_preproc_path",
        default=PREPROC_TEXTS,
        type=str,
        help="The output path for preprocessed corpus. Default: %(default)s.",
    )
    parser.add_argument(
        "--out_unpreproc_path",
        default=UNPREPROC_TEXTS,
        type=str,
        help="The output path for unpreprocessed corpus. Default: %(default)s.",
    )
    parser.add_argument(
        "--valid_upos",
        nargs="+",
        default=["ADJ", "NOUN", "PROPN"],
        help="Part-of-speech to be kept after preprocessing. Default: %(default)s.",
    )
    parser.add_argument(
        "--stopwords",
        nargs="+",
        default=SVEVO_STOPWORDS,
        help="Additional stopwords to be used during preprocessing. Default: %(default)s.",
    )
    parser.add_argument(
        "--no_below",
        default=5,
        type=int,
        help="Lower threshold for word to be kept in preprocessing, in # of occurrences. Default: %(default)s."
    )
    parser.add_argument(
        "--no_above",
        default=0.5,
        type=float,
        help="Upper threshold for word to be kept in preprocessing, in percentage of total occurrences. Default: %(default)s."
    )
    parser.add_argument(
        "--language",
        default="it",
        type=str,
        help="Language used by spaCy preprocessing. Default: %(default)s."
    )
    parser.add_argument(
        "--n_jobs",
        default=-1,
        type=int,
        help="Defines the number of jobs used for multiprocessing. Default: %(default)s (use all available cores)."
    )
    parser.add_argument(
        "--chunksize",
        default=100,
        type=int,
        help="Defines the chunk size used in multiprocessing. Default: %(default)s."
    )
    args = parser.parse_args()
    main(args)