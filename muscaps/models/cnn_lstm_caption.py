import torch
import torch.nn as nn

from muscaps.modules.decoders import LSTMDecoder
from muscaps.modules.encoders import CNNEncoder, WordEmbeddingEncoder, LSTMEncoder


class CNNLSTMCaption(nn.Module):
    def __init__(self, config, vocab, device, teacher_forcing):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.teacher_forcing = teacher_forcing
        self.hidden_dim_encoder = self.config.hidden_dim_encoder
        self.hidden_dim_decoder = self.config.hidden_dim_decoder
        self.fusion = self.config.fusion
        self.finetune = self.config.finetune

        self.audio_encoder = CNNEncoder(self.config, device)
        self.audio_linear = nn.Linear(self.audio_encoder.audio_feature_dim,
                                      self.hidden_dim_encoder)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim_encoder)
        self.relu = nn.ReLU()

        self.text_encoder = WordEmbeddingEncoder(self.vocab)

        self.encoder = LSTMEncoder(self.config, device)
        if self.fusion == "early":
            self.decoder = LSTMDecoder(
                self.config, self.hidden_dim_encoder, self.hidden_dim_decoder)
        elif self.fusion == "late":
            self.decoder = LSTMDecoder(
                self.config, 2 * self.hidden_dim_encoder, self.hidden_dim_decoder)
        self.decoder.init_weights(self.text_encoder.word_embeddings)

    def forward(self, audio, audio_len, x, cap_len):
        audio_features, _ = self.audio_encoder(audio, audio_len)
        audio_embeds = self.audio_linear(audio_features)
        if len(audio_embeds.size()) != 2:
            audio_embeds = audio_embeds.unsqueeze(0)
        audio_embeds = self.relu(self.batch_norm(audio_embeds))

        word_embeds = self.text_encoder(x)

        encoder_out = self.encoder(audio_embeds, word_embeds, cap_len)

        batch_size, seq_len, _ = word_embeds.size()

        # if inference phase, don't use teacher forcing
        if not self.teacher_forcing:
            seq_len += 1
        outputs = torch.zeros(batch_size, seq_len,
                              self.vocab.size).cuda()

        for t in range(seq_len):
            if t == 0:
                h = audio_embeds
                c = audio_embeds
            elif t > 0:
                out, h, c = self.decoder(encoder_out[:, t - 1, :], h, c)
                outputs[:, t, :] = out
        return outputs[:, 1:, :]

    @classmethod
    def config_path(cls):
        return "configs/models/cnn_lstm_caption.yaml"
