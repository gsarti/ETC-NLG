import os
import torch
import multiprocessing as mp
from contextualized_topic_models.models.ctm import CTM

mode_dict = {
    "ContextualInferenceNetwork" : "contextual",
    "CombinedInferenceNetwork" : "combined"
}

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