import torch.nn as nn

from muscaps.utils.utils import masked_softmax


class Attention(nn.Module):
    def __init__(self, config, audio_feature_dim):
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.attention_dim = config.attention_dim
        self.hidden_dim = config.hidden_dim_encoder
        self.attention_type = config.attention_type

        self.audio_projection = nn.Linear(self.audio_feature_dim,
                                          self.attention_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.attention_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.linear = nn.Linear(self.attention_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, hidden_state, audio_features, audio_feature_mask):
        """ 
        Args:
            - hidden_state: Tensor (batch_size, hidden_dim_encoder)
            - audio_features: Tensor of shape (batch_size, num_chunks, audio_feature_dim)
            - audio_feature_mask: Dict[str, torch.Tensor], with Tensor of shape (batch_size, num_chunks, audio_feature_dim)
        """
        projected_features = self.audio_projection(audio_features)
        projected_query = self.query_projection(hidden_state)
        # (batch_size, num_chunks, attention_dim)
        projected_query = projected_query.unsqueeze(1).repeat(
            1, projected_features.size(1), 1)

        if self.attention_type == "mlp":
            attention_scores = self.linear(
                self.tanh(projected_features + projected_query))
            attention_scores = attention_scores.squeeze(-1)
        elif self.linear == "dot":
            joint_repr = projected_features * projected_query
            joint_repr = self.dropout(joint_repr)
            attention_scores = self.linear(joint_repr)

        alpha_weights = masked_softmax(attention_scores,
                                       audio_feature_mask,
                                       dim=-1)

        return alpha_weights
