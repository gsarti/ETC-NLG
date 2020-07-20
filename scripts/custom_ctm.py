import os
import torch
import numpy as np
import logging
import multiprocessing as mp
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.datasets.dataset import CTMDataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)

mode_dict = {
    "ContextualInferenceNetwork" : "contextual",
    "CombinedInferenceNetwork" : "combined"
}


def get_bag_of_words_nofilter(data, min_length):
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length) for x in data]
    return np.array(vect)


class CustomCTM(CTM):
    """ Change format of saved models to make it more sane """
    def __init__(self, input_size, bert_input_size, inference_type, id_name, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, num_data_loader_workers=mp.cpu_count()):
        super(CustomCTM, self).__init__(input_size, bert_input_size, inference_type, n_components,
            model_type, hidden_sizes, activation, dropout, learn_priors, batch_size, lr, momentum,
            solver, num_epochs, reduce_on_plateau, num_data_loader_workers)
        self.identifier = id_name
        
    def _format_file(self):
        epoch = self.nn_epoch + 1
        mode = mode_dict[type(self.model.inf_net).__name__]
        return f"ctm_{self.identifier}_{self.n_components}_{epoch}_{mode}"

    def save(self, models_dir=None):
        if (self.model is not None) and (models_dir is not None):
            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))
            fileloc = os.path.join(models_dir, model_dir, "dicts.pth")
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)


class CustomTextHandler(TextHandler):
    """ Use bag of word without filtering out empty bags """
    def prepare(self):
        data = self.load_text_file()

        concatenate_text = ""
        for line in data:
            line = line.strip()
            concatenate_text += line + " "
        concatenate_text = concatenate_text.strip()

        self.vocab = list(set(concatenate_text.split()))

        for index, vocab in list(zip(range(0, len(self.vocab)), self.vocab)):
            self.vocab_dict[vocab] = index

        self.index_dd = np.array(list(map(lambda y: np.array(list(map(lambda x:
                                                                      self.vocab_dict[x], y.split()))), data)))
        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = get_bag_of_words_nofilter(self.index_dd, len(self.vocab))


class CustomCTMDataset(CTMDataset):
    """ Add filtering for BOW and embeddings if BOW has only 0s """
    def __init__(self, X, X_bert, idx2token, filter_empty_bow=True):
        super(CustomCTMDataset, self).__init__(X, X_bert, idx2token)
        if filter_empty_bow:
            newX, newX_bert = [], []
            for bow, embed in zip(X, X_bert):
                if np.sum(bow) == 0:
                    continue
                newX.append(bow)
                newX_bert.append(embed)
            self.X = np.array(newX)
            self.X_bert = np.array(newX_bert)
            logger.info(f"Shape after filtering: {self.X.shape}, {self.X_bert.shape}")
    
