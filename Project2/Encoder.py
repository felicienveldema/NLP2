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


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, english, hidden):
        english_embed = self.embedding(english)
        output, hidden = self.gru(english_embed, hidden)
        return output, hidden

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(batch_size, 1, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(batch_size, 1, self.hidden_size))