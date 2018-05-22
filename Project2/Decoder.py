from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class BiLinearAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BiLinearAttention, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.normaliser = np.sqrt(hidden_size)
        # Save weights for visualisation
        self.last_weights = None

    def forward(self, english, hidden):
        hidden = hidden.transpose(0, 1).transpose(1,2)
        english = self.w(english)
        weights = self.softmax(torch.bmm(english, hidden) / self.normaliser).squeeze(2)
        self.last_weights = weights.data[0, :]
        return torch.bmm(weights.unsqueeze(1), english)


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.normaliser = np.sqrt(hidden_size)
        # Save weights for visualisation
        self.last_weights = None

    def forward(self, english, hidden):
        hidden = hidden.transpose(0, 1).transpose(1, 2)
        dotproduct = torch.bmm(english, hidden).squeeze(2)
        dotproduct = dotproduct / self.normaliser
        weights = self.softmax(dotproduct)
        self.last_weights = weights.data[0, :]
        return torch.bmm(weights.unsqueeze(1), english)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, end_token, max_length, type, attention_type):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token
        self.type = type

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embedding = nn.Embedding(output_size, hidden_size)
        if self.type == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)

        if attention_type == "bilinear":
            self.attention = BiLinearAttention(hidden_size)
        else:
            self.attention = ScaledDotAttention(hidden_size)
        self.attention_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, french, english, hidden, validation=False):
        # Apply attention to English sentence
        context = self.attention(english, hidden).transpose(0, 1)
        hidden = self.attention_combined(torch.cat((hidden, context), dim=2))

        french = self.embedding(french).unsqueeze(1)
        if not validation:
            french = F.dropout(french, p=self.dropout_rate)
        output, hidden = self.network(french, hidden)
        output_over_vocab = self.out(output[:, 0, :])
        vocab_probs = self.logsoftmax(output_over_vocab)
        return vocab_probs, hidden

    def eval(self, french, english, hidden):
        probs, hidden = self.forward( french, english, hidden, True) 
        weights = self.attention.last_weights
        return probs, hidden, weights

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))