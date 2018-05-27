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
from Attention import MultiHeadAttention


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        return y

class TransformerSublayer(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(TransformerSublayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feedforward1 = nn.Linear(hidden_size, hidden_size * 4)
        self.feedforward2 = nn.Linear(hidden_size * 4, hidden_size)
        self.normalization = LayerNorm(hidden_size)

    def forward(self, english, dropout_rate, validation=False):
        queries = torch.mean(english, dim=1).unsqueeze(0)
        attention =  self.attention(english, english, validation, dropout_rate)
        modified = torch.add(english, attention)
        ff_output = self.feedforward2(F.relu(self.feedforward1(modified)))
        return attention, torch.add(modified, ff_output)

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, input_size, max_pos, enable_cuda):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.type = type
        self.enable_cuda = enable_cuda

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.position_embeddings = nn.Embedding(100, hidden_size)

        self.layer1 = TransformerSublayer(hidden_size, input_size)
        self.layer2 = TransformerSublayer(hidden_size, input_size)
        self.layer3 = TransformerSublayer(hidden_size, input_size)


    def forward(self, english, positions, validation=False):
        english_embed = self.embeddings(english)
        pos_embed = self.position_embeddings(positions)

        if not validation:
            english_embed = F.dropout(english_embed, p=self.dropout_rate)
            pos_embed = F.dropout(pos_embed, p=self.dropout_rate)

        english = torch.add(english_embed, pos_embed)
        hidden, output1 = self.layer1(english, self.dropout_rate, validation)
        hidden, output2 = self.layer2(output1, self.dropout_rate, validation)
        # hidden, output3 = self.layer3(output2, self.dropout_rate, validation)
        return torch.mean(hidden, dim=1).unsqueeze(0), output2

    def eval(self, english, positions):
        return self.forward(english, positions, True)

class Encoder(nn.Module):
    def __init__(self, hidden_size, input_size, max_pos, type, enable_cuda):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.type = type
        self.enable_cuda = enable_cuda

        if self.type.lower() == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True,
                                  bidirectional=True)
        elif self.type.lower() == 'rnn':
            self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.position_embeddings = nn.Embedding(100, hidden_size)

    def forward(self, english, positions, validation=False):
        hidden = self.init_hidden(english.shape[0], self.enable_cuda)
        english_embed = self.embeddings(english)
        pos_embed = self.position_embeddings(positions)

        if not validation:
            english_embed = F.dropout(english_embed, p=self.dropout_rate)
            pos_embed = F.dropout(pos_embed, p=self.dropout_rate)

        english = torch.add(english_embed, pos_embed)
        if self.type.lower() == 'avg':
            return (torch.mean(english, dim=1).unsqueeze(0), english)
        else:
            output, hidden = self.network(english, hidden)
            return (torch.add(hidden[0, :, :], hidden[1, :, :]).unsqueeze(0),
                    torch.add(output[:, :, :self.hidden_size],
                             output[:, :, self.hidden_size:]))

    def eval(self, english, positions):
        return self.forward(english, positions, True)

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(2, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(2, batch_size, self.hidden_size))