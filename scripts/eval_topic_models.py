import os
import sys
import logging
import numpy as np
from shutil import rmtree
from prettytable import PrettyTable
from tqdm import tqdm

from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.evaluation.measures import (
    TopicDiversity,
    CoherenceNPMI,
    InvertedRBO
)

sys.path.append(os.getcwd())

from sentence_transformers import SentenceTransformer
from scripts.sent_transformers import CamemBERT, Pooling

PREPROC_TEXTS = 'data/preprocessed_texts.txt'
UNPREPROC_TEXTS = 'data/unpreprocessed_texts.txt'
UMBERTO_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
UMBERTO_SIZE = 768
SAVE_PATH = 'models'

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def embeddings_from_file(file, model_name):
    we_model = CamemBERT(model_name)
    pooling = Pooling(we_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[we_model, pooling])
    with open(file) as file:
        train_text = list(map(lambda x: x, file.readlines()))
        return np.array(model.encode(train_text))


def main():
    handler = TextHandler(PREPROC_TEXTS)
    handler.prepare()

    train_embeds = embeddings_from_file(UNPREPROC_TEXTS, UMBERTO_NAME)
    train_data = CTMDataset(handler.bow, train_embeds, handler.idx2token)
    x = PrettyTable()
    x.field_names = [
        "Inference Type", "# Topics", "TopicDiversity", "InvertedRBO", "CoherenceNPMI"
    ]
    best_npmi = 0
    for inf_type in ['contextual', 'combined']:
        for n_topics in tqdm(range(3, 11)):
            ctm = CTM(
                input_size=len(handler.vocab),
                bert_input_size=UMBERTO_SIZE,
                hidden_sizes=(100,100,100),
                inference_type=inf_type,
                n_components=n_topics,
                num_epochs=200
            )
            ctm.fit(train_data)
            td = TopicDiversity(ctm.get_topic_lists(25))
            rbo = InvertedRBO(ctm.get_topic_lists(10))
            td_score = td.score(topk=25)
            rbo_score = rbo.score()
            with open(PREPROC_TEXTS, 'r') as f:
                texts = [doc.split() for doc in f.read().splitlines()]
            npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
            npmi_score = npmi.score()
            x.add_row([inf_type, n_topics, td_score, rbo_score, npmi_score])
            if npmi_score > best_npmi:
                logger.info(f"New best NPMI with type {inf_type}, {n_topics} topics: {npmi_score}")
                rmtree(SAVE_PATH)
                ctm.save(models_dir=SAVE_PATH)
                best_type, best_topic = inf_type, n_topics
    logger.info(f"\n{x}")
    logger.info(f"Best model: {inf_type} with {n_topics} topics. Saved to {SAVE_PATH}.")
            
            
if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    main()
