import os
import json
import numpy as np
import torch

from muscaps.utils.vocab import Vocabulary
from muscaps.datasets.base_dataset import BaseDataset


class AudioCaptionDataset(BaseDataset):
    def __init__(self, config, dataset_type="train"):
        super().__init__(config, dataset_type, dataset_name="audiocaption")

        self.dataset_json = os.path.join(
            self._data_dir, "dataset_{}.json".format(self._dataset_type))
        self._load()

    def _load(self):
        self.sr = self.config.audio.sr

        with open(self.dataset_json) as f:
            self.samples = json.load(f)
            self._build_vocab()

            self.captions = [[self.vocab.SOS_TOKEN] + i["caption"]['tokens'] +
                             [self.vocab.EOS_TOKEN] for i in self.samples]

            self.audio_dir = os.path.join(self._data_dir, "audio")
            self.audio_paths = [os.path.join(
                self.audio_dir, i['track_id']+".npy") for i in self.samples]

    def _build_vocab(self):
        """ Build vocab based on captions in the training set"""
        if self._dataset_type == "train":
            self.vocab = Vocabulary(
                [i["caption"]['tokens'] for i in self.samples])
        else:
            training_set = os.path.join(self._data_dir, "dataset_train.json")
            with open(training_set) as f:
                samples = json.load(f)
                training_captions = [i["caption"]['tokens'] for i in samples]
            self.vocab = Vocabulary(training_captions)

    def get_caption(self, idx):
        """Get caption and convert list of strings to tensor of word indices"""
        tokenized_caption = self.captions[idx]
        token_idx = torch.ShortTensor([
            self.vocab.word2idx.get(token, self.vocab.UNK_INDEX)
            for token in tokenized_caption
        ])
        return token_idx

    def get_audio(self, idx):
        audio = np.load(self.audio_paths[idx]).astype('float32')
        audio = torch.Tensor(audio)
        return audio

    def __getitem__(self, idx):
        """Return one data pair (audio, caption)."""
        audio = self.get_audio(idx)
        token_idx = self.get_caption(idx)
        return audio, token_idx

    def __len__(self):
        return len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/dataset.yaml"
