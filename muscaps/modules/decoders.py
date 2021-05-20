import torch.nn as nn
from torch.nn.utils import weight_norm


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_classes = self.config.vocab_size
        self.dropout_p = self.config.dropout_decoder

        self.build()

    def build(self):
        raise NotImplementedError


class LSTMDecoder(Decoder):
    def __init__(self, config, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        super().__init__(config)

    def build(self):
        self.lstm_cell = nn.LSTMCell(input_size=self.input_dim,
                                     hidden_size=self.hidden_dim)
        self.dense = weight_norm(nn.Linear(self.hidden_dim, self.n_classes))
        self.dropout = nn.Dropout(p=self.dropout_p)
        # self.init_weights()

    def init_weights(self, layer=None):
        if layer is None:
            self.dense.bias.data.fill_(0)
            self.dense.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h, c):
        hidden_state, cell_state = self.lstm_cell(x, (h, c))
        out = self.dense(self.dropout(hidden_state))
        return out, hidden_state, cell_state
