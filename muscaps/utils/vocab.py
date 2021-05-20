"""Adapted from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/vocab.py"""

import torch
import os
from torchtext import vocab
from collections import Counter


from muscaps.utils.utils import get_root_dir


class Vocabulary():
    """ Vocabulary class to covert words and numerical indices.

    Attributes:
        token_freq (Counter): frequencies of tokens in the data used to build the Vocab.
        word2idx (defaultdict): mapping of token strings to numerical identifiers.
        idx2word (list): string tokens indexed by their numerical identifiers.
    """

    def __init__(self,
                 tokens=None,
                 token_freq=None,
                 min_count=2,
                 pretrained_emb="glove.6B.300d"):
        """
        Args:
            - tokens: list of string tokens with the input text to create the Vocab from.
            - min_count: threshold token occurrence below which the token is not included in the dictionary
            - pretrained_emb:
        """
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        self.min_count = min_count
        self.pretrained_emb = pretrained_emb

        if token_freq is None:
            self.token_freq = Counter(
                [i for token_sublist in tokens for i in token_sublist])
        else:
            self.token_freq = token_freq

        self.token_list = []
        for token in self.token_freq:
            if self.token_freq[token] >= min_count:
                self.token_list.append(token)

        extras = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN
        ]
        self.idx2word = extras + self.token_list
        self._build_vocab()
        self._load_vectors(vector_cache=os.path.join(get_root_dir(), "data"))

    def _build_vocab(self):
        self.word2idx = {v: k for k, v in enumerate(self.idx2word)}

        self.size = self.get_size()

        self.UNK_INDEX = (self.word2idx[self.UNK_TOKEN]
                          if self.UNK_TOKEN in self.word2idx else None)

        self.PAD_INDEX = (self.word2idx[self.PAD_TOKEN]
                          if self.PAD_TOKEN in self.word2idx else None)

    def _load_vectors(self, vector_cache):
        if self.pretrained_emb not in vocab.pretrained_aliases:
            error = "Unknown embedding type: %s" % self.pretrained_emb, "error"
            raise RuntimeError(error)

        if os.path.exists(vector_cache):
            embedding = vocab.pretrained_aliases[self.pretrained_emb](
                cache=vector_cache)
        else:
            embedding = vocab.pretrained_aliases[self.pretrained_emb]()
        # dim of vectors: V x K (vocab dim x embedding dim)
        self.vectors = torch.FloatTensor(self.size, embedding.dim)

        for i in range(4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.size):
            word = self.idx2word[i]
            embedding_index = embedding.stoi.get(word, None)

            if embedding_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX]
            else:
                self.vectors[i] = embedding.vectors[embedding_index]

    def get_id(self, token):
        """Convert a string tokens to its numerical id."""
        if token in self.word2idx:
            return self.word2idx[token]
        else:
            return self.UNK_INDEX

    def get_string(self, idx):
        """Convert a numerical id to its string token."""
        return self.idx2word[idx]

    def get_size(self):
        """Get total number of tokens in the vocabulary."""
        return len(self.idx2word)

    def get_embedding(self, word):
        return self.vectors[self.word2idx[word]]

    def save(self, path):
        """Save `self.idx2word` in `path`."""
        with open(path, "w") as f:
            f.write("\n".join(self.idx2word))

    def load(self):
        """Load vocab txt file from `vocab_file`."""
        with open(self.vocab_file) as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        return lines
