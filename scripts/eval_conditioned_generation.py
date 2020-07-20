""" Produce the confusion matrix between conditioning labels used for PPLM generation (gold) and prediction on
generated sequences performed by the same topic model used for training the conditioning discriminator. We use
this as a measure of how coherent is generation w.r.t the conditioning topic, regardless of topic meaningfulness."""

import os
import sys
import logging
import argparse
import spacy
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

from scripts.custom_ctm import get_ctm_and_data
from scripts.preprocess import preprocess_texts

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

PREPROC_TEXTS = 'data/preprocessed_svevo_texts.txt'
MODEL_DIR_CONTEXTUAL = "models/ctm_svevo_6_100_contextual"
MODEL_DIR_COMBINED = "models/ctm_svevo_6_100_combined"
EMBEDS_PATH = "models/preprocessed_svevo_texts_umberto-commoncrawl-cased-v1"
GEN_TEXT_PATH = "tests/Svevo/fine_tuned_LM_blockSize=128_ep=2/discriminator_ep=10_contextual_1500/generated_text_labels=contextual_samples=3.csv"
SVEVO_STOPWORDS = [
    'schmitz', 'signore', 'signora', 'mano', 'ettore', 'lettera', 'parola', 'fare', 'cosa'
]

def main(args):
    gen_data = pd.read_csv(args.gen_text_path)
    args.nlp = spacy.load(args.language)
    preproc_texts = [" ".join(x) for x in preprocess_texts(args, gen_data['perturbed_gen_text'])]
    ctm, data = get_ctm_and_data(args.preproc_path, args.embeds_path, False, args.model_dir, args.inference_type, data=preproc_texts)
    dist = ctm.get_thetas(data)
    logger.info(f"Thetas shape: ({len(dist)},{len(dist[0])})")
    topics = ctm.get_topic_lists(5)
    best_topics = np.argmax(dist, axis=1)
    gen_data["class_pred"] = ["|".join(topics[i]) for i in best_topics]
    lab, pred = gen_data["class_label"], gen_data["class_pred"]
    cm_labs = list(set(lab) | set(pred))
    logger.info("Confusion matrix: f{confusion_matrix(lab, pred, labels=cm_labs)}")
    gen_data.to_csv(f'{args.gen_text_path.split(".")[0]}_predicted.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_text_path",
        default=GEN_TEXT_PATH,
        type=str,
        help="The path where generated texts can be found. Default: %(default)s."
        "The generated text file should be a csv with 3 columns, cond_text containing LM input for generation "
        "(e.g. 'It's possible'), perturbed_gen_text containing the generated text and class_label containing the "
        "top 5 words of the topic used for conditioned generation, separated by |. The script run_pplm.sh produces "
        "files in this format",
    )
    parser.add_argument(
        "--preproc_path",
        default=PREPROC_TEXTS,
        type=str,
        help="Preprocessed text path, used to generate the vocabulary for the new data. Default: %(default)s.",
    )
    parser.add_argument(
        "--embeds_path",
        default=EMBEDS_PATH,
        type=str,
        help="Path to cached embeddings. Default: %(default)s.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="The directory from which the saved model should be loaded. Default: %(default)s.",
    )
    parser.add_argument(
        "--inference_type",
        type=str,
        choices=["contextual", "combined"],
        default="contextual",
        help="Topic modeling mode of the loaded model. One between: %(choice)s. Default: %(default)s.",
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
        "--language",
        default="it",
        type=str,
        help="Language used by spaCy preprocessing. Default: %(default)s."
    )
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = MODEL_DIR_CONTEXTUAL if args.inference_type == "contextual" else MODEL_DIR_COMBINED
    logger.info(f"Script args: f{args}")
    main(args)