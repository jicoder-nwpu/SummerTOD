from torch import nn


class Adaptor(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, inner_dim=1024):
        super().__init__()
        self.d_model = d_model
        self.inner_dim = inner_dim
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.wi = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.wo = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, relu=False):
        normed_hidden_states = self.layer_norm(hidden_states)

        inner_hidden_states = self.wi(normed_hidden_states)

        if relu:
            inner_hidden_states = nn.functional.relu(inner_hidden_states)

        inner_hidden_states = self.wo(inner_hidden_states)
        hidden_states = hidden_states + self.dropout(inner_hidden_states)

        return hidden_states