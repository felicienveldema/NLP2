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
    def __init__(self, hidden_size, output_size, end_token):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(output_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        print(output[:, 0, :].shape)
        output_over_vocab = self.out(output[:, 0, :])
        print(self.out.weight.data.shape)
        print(output_over_vocab.shape)
        vocab_probs = self.softmax(output_over_vocab)
        return vocab_probs, hidden

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(batch_size, 1, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(batch_size, 1, self.hidden_size))