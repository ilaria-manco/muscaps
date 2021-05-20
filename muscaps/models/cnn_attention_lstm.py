import torch
import torch.nn as nn

from muscaps.modules.decoders import LSTMDecoder
from muscaps.modules.attention import Attention
from muscaps.modules.encoders import CNNEncoder, WordEmbeddingEncoder


class AttentionModel(nn.Module):
    def __init__(self, config, vocab, device, teacher_forcing):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.teacher_forcing = teacher_forcing
        self.attention_dim = self.config.attention_dim
        self.hidden_dim_encoder = self.config.hidden_dim_encoder
        self.hidden_dim_decoder = self.config.hidden_dim_decoder
        self.finetune = self.config.finetune

        self.audio_encoder = CNNEncoder(self.config, device)
        self.audio_feature_dim = self.audio_encoder.audio_feature_dim
        self.text_encoder = WordEmbeddingEncoder(self.vocab)
        self.word_embed_dim = self.text_encoder.word_embed_dim

        self.audio_linear = nn.Linear(self.audio_encoder.audio_feature_dim,
                                      self.hidden_dim_encoder)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim_encoder)
        self.relu = nn.ReLU()
        self.encoder = nn.LSTMCell(self.word_embed_dim +
                                   2 * self.hidden_dim_encoder, self.hidden_dim_encoder)
        self.attention = Attention(self.config, self.audio_feature_dim)
        self.decoder = LSTMDecoder(
            self.config, self.hidden_dim_decoder + self.audio_feature_dim, self.hidden_dim_decoder)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_lstm_states(self, features):
        state = features
        lstm_states = {
            "h1": state.clone(),
            "c1": state.clone(),
            "h2": state.clone(),
            "c2": state.clone(),
        }
        return lstm_states

    def forward(self, audio, audio_len, caption=None, states=None):
        pooled_features, audio_features = self.audio_encoder(audio, audio_len)
        pooled_audio_embeds = self.audio_linear(pooled_features)
        if len(pooled_audio_embeds.size()) != 2:
            pooled_audio_embeds = pooled_audio_embeds.unsqueeze(0)
        pooled_audio_embeds = self.relu(self.batch_norm(pooled_audio_embeds))

        word_embeds = self.text_encoder(caption)

        self.batch_size, seq_len, _ = word_embeds.size()

        # Create mask for audio features (to be used in softmax later)
        max_len = audio_features.size(1)
        audio_len = audio_len // self.audio_encoder.input_length
        audio_feature_mask = torch.arange(max_len).expand(
            len(audio_len), max_len) < audio_len.unsqueeze(1)
        audio_feature_mask = audio_feature_mask.cuda()

        states = self.init_lstm_states(pooled_audio_embeds)

        if self.teacher_forcing:
            seq_len -= 1

        outputs = torch.zeros(self.batch_size, seq_len, self.vocab.size).cuda()

        if len(pooled_audio_embeds.size()) == 1:
            pooled_audio_embeds = pooled_audio_embeds.unsqueeze(0)

        for t in range(seq_len):
            # ENCODER
            word_embed = word_embeds[:, t, :]
            encoder_lstm_input = torch.cat(
                [word_embed, pooled_audio_embeds, states["h2"]], dim=1)
            states["h1"], states["c1"] = self.encoder(
                encoder_lstm_input, (states["h1"], states["c1"]))

            # ATTENTION
            attention_weights = self.attention(states["h1"], audio_features,
                                               audio_feature_mask)
            attended_features = torch.sum(attention_weights.unsqueeze(-1) *
                                          audio_features,
                                          dim=1)

            # DECODER
            decoder_input = torch.cat([attended_features, states["h1"]], dim=1)
            out, states["h2"], states["c2"] = self.decoder(
                decoder_input, states["h2"], states["c2"])

            outputs[:, t, :] = out

        return outputs

    @classmethod
    def config_path(cls):
        return "configs/models/cnn_attention_lstm.yaml"
