import torch.nn as nn
import copy

from source.model.attention_model import MultiHeadAttention
from source.model.embeddings import Embeddings
from source.model.embeddings import PositionalEncoding
from source.model.encoder import Encoder
from source.model.encoder import EncoderLayer
from source.model.bert_model import BERTClassification
from source.model.bert_model import PositionwiseFeedForward


def make_model(src_vocab,
               num_classes,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = \
        BERTClassification(
                            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                            num_classes
        )

    # Initialize parameters with Glorot
    for p in model.paremeters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_main():
    bert_model = make_model()


if __name__ == '__main__':
    run_main()
