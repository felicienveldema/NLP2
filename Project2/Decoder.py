from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
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

    def forward(self, english, hidden):
        hidden = hidden.transpose(0, 1).transpose(1,2)
        english = self.w(english)
        weights = self.softmax(torch.bmm(english, hidden) / self.normaliser).squeeze(2)
        return torch.bmm(weights.unsqueeze(1), english)


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.normaliser = np.sqrt(hidden_size)

    def forward(self, english, hidden):
        hidden = hidden.transpose(0, 1).transpose(1, 2)
        dotproduct = torch.bmm(english, hidden).squeeze(2)
        dotproduct = dotproduct / self.normaliser
        weights = self.softmax(dotproduct)
        return torch.bmm(weights.unsqueeze(1), english)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, end_token, max_length, type, attention_type):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token
        self.type = type

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


    def forward(self, french, english, hidden):
        # Apply attention to English sentence
        english_attention = self.attention(english, hidden)

        french = self.embedding(french).unsqueeze(1)
        a = torch.cat((french, english_attention), dim=2)
        network_input = self.attention_combined(a)
        output, hidden = self.network(network_input, hidden)
        output_over_vocab = self.out(output[:, 0, :])
        vocab_probs = self.logsoftmax(output_over_vocab)
        return vocab_probs, hidden



    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))