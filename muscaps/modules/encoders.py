import random
import torch
import torch.nn as nn

from muscaps.modules.audio_feature_extractors import HarmonicCNN, Musicnn


class AudioEncoder(nn.Module):
    """ Base class for audio encoders."""

    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.config = config
        self.pool_type = self.config.pool_type

        self.build()

    def build(self):
        raise NotImplementedError


class CNNEncoder(AudioEncoder):
    def __init__(self, config, device):
        super().__init__(config, device)

    def build(self):
        self._build_feature_extractor()

    def _build_feature_extractor(self):
        self.feature_extractor_type = self.config.feature_extractor_type
        self.feature_extractor_path = self.config.feature_extractor_path
        self.pretrained_version = self.config.pretrained_version
        if self.feature_extractor_type == "hcnn":
            self.feature_extractor = HarmonicCNN()
            self.input_length = 5 * 16000
            self.audio_feature_dim = 256
        elif self.feature_extractor_type == "musicnn":
            self.feature_extractor = Musicnn(
                dataset=self.pretrained_version)
            self.input_length = 3 * 16000
            self.audio_feature_dim = 2097 if self.pretrained_version == "msd" else 753
        if self.pretrained_version is not None:
            state_dict = torch.load(self.feature_extractor_path)
            self.feature_extractor.load_state_dict(
                state_dict, strict=False)
        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, self.audio_feature_dim))

    def extract_features(self, audio, audio_len):
        audio_chunks = torch.split(audio, self.input_length, 1)
        audio_chunks = torch.stack(
            [i for i in audio_chunks if i.size(1) == self.input_length],
            dim=0)
        if self.pretrained_version is None:
            num_chunks_to_select = 4
            max_non_zero_len = int(min(audio_len).item(
            )/(self.input_length)) - num_chunks_to_select
            random_int_start = random.randint(0, max_non_zero_len)
            audio_chunks = audio_chunks[random_int_start:
                                        random_int_start+num_chunks_to_select]

        num_chunks, batch_size, _ = audio_chunks.size()
        audio_features = torch.zeros(batch_size, num_chunks,
                                     self.audio_feature_dim)

        for i, chunk in enumerate(audio_chunks):
            if torch.nonzero(chunk).size(1) != 0:
                if self.feature_extractor_type == "hcnn":
                    audio_features[:, i] = self.feature_extractor(chunk)
                elif self.feature_extractor_type == "musicnn":
                    feature = self.feature_extractor(chunk)
                    feature = nn.MaxPool1d(feature.size(-1))(feature)
                    feature = feature.squeeze(-1)
                    audio_features[:, i] = feature
        audio_features = audio_features.to(self.device)
        return audio_features

    def forward(self, x, audio_len):
        x = self.extract_features(x, audio_len)
        if self.pool_type == "avg":
            pooled_x = self.pool(x)
            pooled_x = pooled_x.squeeze()
        else:
            pooled_x = x
        return pooled_x, x


class TextEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        self.build()

    def build(self):
        raise NotImplementedError


class WordEmbeddingEncoder(TextEncoder):
    def __init__(self, vocab):
        super().__init__(vocab)

    def build(self):
        self._build_word_embedding()

    def _build_word_embedding(self):
        glove_weights = self.vocab.vectors
        self.word_embeddings = nn.Embedding.from_pretrained(
            glove_weights, padding_idx=self.vocab.PAD_INDEX)
        self.word_embed_dim = self.word_embeddings.embedding_dim

    def forward(self, x):
        word_embeds = self.word_embeddings(x)

        return word_embeds


class SentenceEncoder(nn.Module):
    """ Base class for sentence encoders."""

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.word_embed_dim = self.config.word_embed_dim
        self.hidden_dim = self.config.hidden_dim_encoder
        self.fusion = self.config.fusion

        self.build()

    def build(self):
        raise NotImplementedError


class LSTMEncoder(SentenceEncoder):
    def __init__(self, config, device):
        super().__init__(config, device)

    def build(self):
        self._encode_sentence()

    def _encode_sentence(self):
        if self.fusion == "init":
            self.lstm = nn.LSTM(self.word_embed_dim,
                                self.hidden_dim,
                                batch_first=True)
        elif self.fusion == "early":
            self.lstm = nn.LSTM(self.word_embed_dim + self.hidden_dim,
                                self.hidden_dim,
                                batch_first=True)
        elif self.fusion == "late":
            self.lstm = nn.LSTM(self.word_embed_dim,
                                self.hidden_dim,
                                batch_first=True)

    def forward(self, audio_embeds, word_embeds, x_len):
        audio_embeds = audio_embeds.unsqueeze(1)
        if len(audio_embeds.size()) == 2:
            audio_embeds = audio_embeds.unsqueeze(0)

        batch_size, _, _ = audio_embeds.size()
        c_init = torch.zeros((1, batch_size, self.hidden_dim)).to(self.device)

        if self.fusion == "init":
            lstm_input = word_embeds
            h_init = audio_embeds
        elif self.fusion == "early":
            h_init = torch.zeros(
                (1, batch_size, self.hidden_dim)).to(self.device)
            audio_embeds = torch.cat(word_embeds.size(1) * [audio_embeds],
                                     dim=1)
            lstm_input = torch.cat((audio_embeds, word_embeds), dim=2)
        elif self.fusion == "late":
            audio_embeds = torch.cat(word_embeds.size(1) * [audio_embeds],
                                     dim=1)
            lstm_input = word_embeds
            h_init = torch.zeros(
                (1, batch_size, self.hidden_dim)).to(self.device)
        x_len = torch.as_tensor(x_len, dtype=torch.int64, device="cpu")
        lstm_input = nn.utils.rnn.pack_padded_sequence(lstm_input,
                                                       x_len,
                                                       enforce_sorted=False,
                                                       batch_first=True)
        lstm_out, (hidden_state, _) = self.lstm(lstm_input, (h_init, c_init))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                             batch_first=True)
        if self.fusion == "late":
            lstm_out = torch.cat((lstm_out, audio_embeds), dim=2)

        return lstm_out
