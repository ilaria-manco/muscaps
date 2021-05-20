import os
import json
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from muscaps.utils.vocab import Vocabulary
from muscaps.datasets.base_dataset import custom_collate_fn
from muscaps.models.cnn_lstm_caption import CNNLSTMCaption
from muscaps.models.cnn_attention_lstm import AttentionModel
from muscaps.datasets.audiocaption import AudioCaptionDataset


class Evaluation:
    def __init__(self, config, logger, experiment_id):
        self.config = config
        self.logger = logger
        self.device = torch.device(self.config.training.device)
        self.experiment_id = experiment_id
        self.path_to_model = os.path.join(self.config.env.experiments_dir,
                                          self.experiment_id,
                                          "best_model.pth.tar")

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        self.logger.write("Loading dataset")
        dataset_name = self.config.dataset_config.dataset_name
        if dataset_name == "audiocaption":
            test_dataset = AudioCaptionDataset(
                self.config.dataset_config, dataset_type="test")
        else:
            raise ValueError(
                "{} dataset is not supported.".format(dataset_name))
        token_freq_dict = json.load(open(self.logger.vocab_path, 'r'))
        self.vocab = Vocabulary(tokens=None, token_freq=token_freq_dict)
        OmegaConf.update(self.config, "model_config.vocab_size",
                         self.vocab.size)
        self.test_loader = DataLoader(
            dataset=test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    def build_model(self):
        self.logger.write("Building model")
        model_name = self.config.model_config.model_name
        if model_name == "cnn_lstm_caption":
            self.model = CNNLSTMCaption(
                self.config.model_config, self.vocab, self.device, teacher_forcing=False)
        elif model_name == "cnn_attention_lstm":
            self.model = AttentionModel(
                self.config.model_config, self.vocab, self.device, teacher_forcing=False)
        else:
            raise ValueError("{} model is not supported.".format(model_name))
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def obtain_predictions(self):
        if os.path.exists(self.predictions_path):
            predictions, true_captions, audio_paths = json.load(
                open(self.predictions_path)).values()
        else:
            print("No captions found.")
        return predictions, true_captions, audio_paths
