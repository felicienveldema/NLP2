from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, end_token, type):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token
        self.type = type

        self.embedding = nn.Embedding(output_size, hidden_size)
        if self.type == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.embedding(input).unsqueeze(1)
        output, hidden = self.network(input, hidden)
        output_over_vocab = self.out(output[:, 0, :])
        vocab_probs = self.softmax(output_over_vocab)
        return vocab_probs, hidden

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))