import torch
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self, config, dataset_type, dataset_name):
        """Base class for a dataset.

        Args:
        - config: dict object with dataset config
        - dataset_type: "train", "test" or "val"
        - dataset_name: string of the dataset name 
        """
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._data_dir = self.config.data_dir

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        __getitem__ of a torch dataset.
        Args:
            idx (int): Index of the sample to be loaded.
        """

        raise NotImplementedError


def custom_collate_fn(data):
    """ Custom collate function for data loader to create mini-batch tensors of the same shape. 

    Args:
    data: list of tuple (audio, caption). 
        - audio: torch tensor of shape (?); variable length.
        - caption: torch tensor of shape (?); variable length.
    Returns:
        padded_audio: torch tensor of shape (batch_size, padded_audio_length).
        padded_captions: torch tensor of shape (batch_size, padded_cap_length).
    """

    data.sort(key=lambda x: len(x[1]), reverse=True)
    audio_tracks, captions = zip(*data)

    audio_lengths = [len(audio) for audio in audio_tracks]
    padded_audio = torch.zeros(
        len(audio_tracks), max(audio_lengths)).float()

    cap_lengths = [len(cap) for cap in captions]
    padded_captions = torch.zeros(len(captions), max(cap_lengths)).long()

    for i, cap in enumerate(captions):
        caption_end = cap_lengths[i]
        padded_captions[i, :caption_end] = cap

        audio_end = audio_lengths[i]
        padded_audio[i, :audio_end] = audio_tracks[i]

    audio_lengths = torch.Tensor(audio_lengths).long()

    return padded_audio, audio_lengths, padded_captions, cap_lengths
