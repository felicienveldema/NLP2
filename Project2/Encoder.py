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
    def __init__(self, hidden_size, input_size, max_pos, type):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.type = type

        if self.type.lower() == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif self.type.lower() == 'rnn':
            self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)

        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)

    def forward(self, english, positions, hidden):
        english_embed = self.embeddings(english)
        pos_embed = self.position_embeddings(positions)
        english = torch.add(english_embed, pos_embed)
        if self.type.lower() == 'avg':
            return torch.mean(english, dim=1).unsqueeze(0), english
        else:
            output, hidden = self.network(english, hidden)
            return hidden, output

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))