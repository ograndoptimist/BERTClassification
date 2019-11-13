import torch.nn as nn
import torch.nn.functional as F


class BERTClassification(nn.Module):
    """
        A Encoder model that has on top feed-forward and softmax to classify text.
    """

    def __init__(self,
                 encoder,
                 src_embed
                 ):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def forward(self, src, src_mask):
        """
            Take in and process masked src sequences.
        """
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


class PositionwiseFeedForward(nn.Module):
    """
        Implements FFN equation.
    """
    def __init__(self,
                 d_model,
                 d_ff,
                 dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
