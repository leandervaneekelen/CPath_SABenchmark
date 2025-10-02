import torch
import torch.nn as nn


class LoopABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.method = "LoopABMIL"

    def forward(self, batch):

        # Save attention/features for each sample
        features = []
        att_weights = []

        # Apply attention to each sample in the batch
        for x in batch:
            x = self.projector(x)
            x = self.activation(x)
            x = self.dropout(x)
            a = self.attention(x)
            att_weights.append(a.view(-1))
            a = torch.softmax(a, dim=0)
            x = torch.sum(a * x, dim=0)
            features.append(x)

        # Stack attention weights and features
        features = torch.stack(features, dim=0)

        # Pass through classifier
        features = self.dropout(features)
        logits = self.classifier(features)
        Y_prob = torch.sigmoid(logits)
        Y_hat = torch.argmax(logits, dim=1)
        results_dict = {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "att_weights": att_weights,
        }
        return results_dict
