import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Attention import ScaledDotAttention, MultiHeadAttention, BiLinearAttention

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, end_token, max_length, type, attention_type):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token
        self.type = type
        self.attention_type = attention_type

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embedding = nn.Embedding(output_size, hidden_size)
        if self.type == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)

        if attention_type.lower() == "bilinear":
            self.attention = BiLinearAttention(hidden_size)
        elif attention_type.lower() == "multihead":
            self.attention = MultiHeadAttention(hidden_size)
        else:
            self.attention = ScaledDotAttention(hidden_size)

        self.attention_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, french, english, hidden, validation=False):
        # Apply attention to English sentence
        context = self.attention(english, hidden, validation, self.dropout_rate).transpose(0, 1)
        hidden = self.attention_combined(torch.cat((hidden, context), dim=2))

        french = self.embedding(french).unsqueeze(1)
        if not validation:
            french = F.dropout(french, p=self.dropout_rate)
            hidden = F.dropout(hidden, p=self.dropout_rate)
        output, hidden = self.network(french, hidden)

        if not validation:
            output_over_vocab = self.out(F.dropout(output[:, 0, :], p=self.dropout_rate))
        else:
            output_over_vocab = self.out(output[:, 0, :])
        vocab_probs = self.logsoftmax(output_over_vocab)
        return vocab_probs, hidden

    def eval(self, french, english, hidden):
        probs, hidden = self.forward( french, english, hidden, True) 
        if self.attention_type.lower() == "multihead":
            weights = [self.attention.last_weights1,self.attention.last_weights2,
                       self.attention.last_weights3] 
        else:
            weights = self.attention.last_weights
        return probs, hidden, weights

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))