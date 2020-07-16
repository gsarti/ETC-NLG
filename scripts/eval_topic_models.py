import os
import sys
import logging
import argparse
import pickle
import torch
import numpy as np
from shutil import rmtree
from prettytable import PrettyTable
from tqdm import tqdm

from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.evaluation.measures import (
    TopicDiversity,
    CoherenceNPMI,
    InvertedRBO
)

sys.path.append(os.getcwd())

from sentence_transformers import SentenceTransformer
from scripts.sent_transformers import CamemBERT, RoBERTa, Pooling
from scripts.custom_ctm import CustomCTM

PREPROC_TEXTS = 'data/preprocessed_svevo_texts.txt'
UNPREPROC_TEXTS = 'data/unpreprocessed_svevo_texts.txt'
UMBERTO_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
UMBERTO_SIZE = 768
SAVE_PATH = 'models'

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def embeddings_from_file(args):
    if args.language == "it":
        we_model = CamemBERT(args.embed_model_name)
    else:
        we_model = RoBERTa(args.embed_model_name)
    pooling = Pooling(we_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[we_model, pooling])
    with open(args.unpreproc_path) as f:
        train_text = list(map(lambda x: x, f.readlines()))
        return np.array(model.encode(train_text))


def main(args):
    handler = TextHandler(args.preproc_path)
    handler.prepare()
    train_embeds = embeddings_from_file(args)
    model_name = args.embed_model_name.split("/")[-1]
    data_name = args.preproc_path.split("/")[-1].split(".")[0]
    with open(os.path.join(args.save_path, f"{data_name}_{model_name}"), 'wb') as f:
        pickle.dump(train_embeds, f)
        logger.info(f"Cached embeddings to {os.path.join(args.save_path, f'{data_name}_{model_name}')}")
    train_data = CTMDataset(handler.bow, train_embeds, handler.idx2token)
    x = PrettyTable()
    x.field_names = [
        "Inference Type", "# Topics", "TopicDiversity", "InvertedRBO", "CoherenceNPMI"
    ]
    best_npmi = 0
    for inf_type in args.modes:
        for n_topics in tqdm(args.n_topics):
            ctm = CustomCTM(
                input_size=len(handler.vocab),
                bert_input_size=args.embed_model_size,
                id_name=args.model_identifier,
                hidden_sizes=(args.hidden_size,args.hidden_size,args.hidden_size),
                inference_type=inf_type,
                n_components=n_topics,
                num_epochs=args.num_epochs
            )
            ctm.fit(train_data)
            td = TopicDiversity(ctm.get_topic_lists(args.diversity_topk))
            rbo = InvertedRBO(ctm.get_topic_lists(args.rbo_topk))
            td_score = td.score(topk=args.diversity_topk)
            rbo_score = rbo.score()
            with open(args.preproc_path, 'r') as f:
                texts = [doc.split() for doc in f.read().splitlines()]
            npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(args.rbo_topk))
            npmi_score = npmi.score()
            x.add_row([inf_type, n_topics, td_score, rbo_score, npmi_score])
            if npmi_score > best_npmi:
                logger.info(f"New best NPMI with type {inf_type}, {n_topics} topics: {npmi_score}")
                best_type, best_topic = inf_type, n_topics
            ctm.save(models_dir=args.save_path)
    logger.info(f"\n{x}")
    logger.info(f"Best model: {inf_type} with {n_topics} topics. Saved to {args.save_path}.")
            
            
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
        "--embed_model_name",
        default=UMBERTO_NAME,
        type=str,
        help="The embedding model name. Must use CamemBERT architecture for italian,"
        "RoBERTa architecture otherwise. Default: %(default)s.",
    )
    parser.add_argument(
        "--model_identifier",
        default="svevo",
        type=str,
        help="Model identifier used when saving the CTM to file. Default: %(default)s.",
    )
    parser.add_argument(
        "--embed_model_size",
        default=UMBERTO_SIZE,
        help="Size of embeddings used in the contextual language model. Default: %(default)s.",
    )
    parser.add_argument(
        "--save_path",
        default=SAVE_PATH,
        type=str,
        help="The save path for the top-scoring topic model. Default: %(default)s.",
    )
    parser.add_argument(
        "--language",
        default="it",
        type=str,
        help="Language of the input, used to determine which language model architecture to use. Default: %(default)s."
    )
    parser.add_argument(
        "--n_topics",
        nargs="+",
        type=int,
        default=[x for x in range(3,11)],
        help="Number of topics to be used in experiments. Default: %(default)s.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="Hidden size of the contextual topic model. Default: %(default)s.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: %(default)s.",
    )
    parser.add_argument(
        "--diversity_topk",
        type=int,
        default=25,
        help="Topic diversity topk parameter. Default: %(default)s.",
    )
    parser.add_argument(
        "--rbo_topk",
        type=int,
        default=10,
        help="InvertedRBO topk parameter. Default: %(default)s.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["contextual", "combined"],
        default=["contextual", "combined"],
        help="Topic modeling mode. One or more between: %(choice)s. Default: %(default)s.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger.info(f"Script args: f{args}")
    main(args)
