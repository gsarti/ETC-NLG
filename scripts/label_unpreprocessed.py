import os
import sys
import logging
import argparse
import pickle
import torch
import numpy as np

from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset

sys.path.append(os.getcwd())

from sentence_transformers import SentenceTransformer
from scripts.sent_transformers import CamemBERT, RoBERTa, Pooling
from scripts.custom_ctm import CustomCTM

PREPROC_TEXTS = 'data/preprocessed_svevo_texts.txt'
UNPREPROC_TEXTS = 'data/unpreprocessed_svevo_texts.txt'
EMBEDS_PATH = "models/preprocessed_svevo_texts_umberto-commoncrawl-cased-v1"
MODEL_DIR_CONTEXTUAL = "models/ctm_svevo_6_200_contextual"
MODEL_DIR_COMBINED = "models/ctm_svevo_6_200_combined"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def load_ctm(args):
    model_file = os.path.join(args.model_dir, "dicts.pth")
    with open(model_file, 'rb') as model_dict:
        checkpoint = torch.load(model_dict)
    ctm = CustomCTM(
        input_size=checkpoint['dcue_dict']['input_size'],
        bert_input_size=checkpoint['dcue_dict']['bert_size'],
        inference_type=args.inference_type,
        id_name=checkpoint['dcue_dict']['id_name'],
        n_components=checkpoint['dcue_dict']['n_components'],
        model_type=checkpoint['dcue_dict']['model_type'],
        hidden_sizes=checkpoint['dcue_dict']['hidden_sizes'],
        activation=checkpoint['dcue_dict']['activation'],
        dropout=checkpoint['dcue_dict']['dropout'],
        learn_priors=checkpoint['dcue_dict']['learn_priors'],
        batch_size=checkpoint['dcue_dict']['batch_size'],
        lr=checkpoint['dcue_dict']['lr'],
        momentum=checkpoint['dcue_dict']['momentum'],
        solver=checkpoint['dcue_dict']['solver'],
        num_epochs=checkpoint['dcue_dict']['num_epochs'],
        reduce_on_plateau=checkpoint['dcue_dict']['reduce_on_plateau'],
        num_data_loader_workers=checkpoint['dcue_dict']['num_data_loader_workers'],
    )
    for (k, v) in checkpoint['dcue_dict'].items():
        setattr(ctm, k, v)
    ctm.model.load_state_dict(checkpoint['state_dict'])
    topics = ctm.get_topic_lists()
    logger.info(f"Loaded {args.inference_type} model at {model_file} with {len(topics)} topics. Showing first 10 topics:")
    for i, topic in enumerate(topics[:10]):
        logger.info(f"Topic {i}: {topic}")
    return ctm


def main(args):
    handler = TextHandler(args.preproc_path)
    handler.prepare() # create vocabulary and training data
    with open(args.embeds_path, 'rb') as f:
        training_embeds = pickle.load(f)
    training_dataset = CTMDataset(handler.bow, training_embeds, handler.idx2token)
    ctm = load_ctm(args.model_dir, args.epoch, args.inference_type)
    dist = ctm.get_thetas(training_dataset)
    logger.info(f"Thetas shape: ({len(dist)},{len(dist[0])})")
    with open(args.unpreproc_path) as f:
        text = list(map(lambda x: x, f.readlines()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preproc_path",
        default=PREPROC_TEXTS,
        type=str,
        help="The output path for preprocessed corpus. Default: %(default)s.",
    )
    parser.add_argument(
        "--unpreproc_path",
        default=UNPREPROC_TEXTS,
        type=str,
        help="The output path for unpreprocessed corpus. Default: %(default)s.",
    )
    parser.add_argument(
        "--embeds_path",
        default=EMBEDS_PATH,
        type=str,
        help="Path to cached embeddings. Default: %(default)s.",
    )
    parser.add_argument(
        "--epoch",
        default=49,
        help="Epoch of the saved model. Default: %(default)s.",
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
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = MODEL_DIR_CONTEXTUAL if args.inference_type == "contextual" else MODEL_DIR_COMBINED
    main(args)
