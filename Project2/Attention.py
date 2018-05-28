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

    def forward(self, english, hidden, validation, dropout_rate):
        """Forward pass of bilinear attention mechanism.

        Args:
            english (Variable FloatTensor): embeddings
            hidden (Variable FloatTensor): last hidden state of decoder
            validation (bool): whether to apply dropout
            dropout_rate (float): how much dropout to apply

        Returns:
            weighted representation of input embeddings
        """
        hidden = hidden.transpose(0, 1).transpose(1,2)
        english = self.w(english)
        weights = self.softmax(torch.bmm(english, hidden) / self.normaliser).squeeze(2)
        if not validation:
            weights = F.dropout(weights, p=dropout_rate)
        self.last_weights = weights.data[0, :]
        return torch.bmm(weights.unsqueeze(1), english)


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.normaliser = np.sqrt(hidden_size)
        # Save weights for visualisation
        self.last_weights = None

    def forward(self, english, hidden, validation, dropout_rate, v=None):
        """Forward pass of scaled dot product mechanism.

        Args:
            english (Variable FloatTensor): embeddings
            hidden (Variable FloatTensor): last hidden state of decoder
            validation (bool): whether to apply dropout
            dropout_rate (float): how much dropout to apply
            v (Variable FloatTensor): the values to apply the weights to

        Returns:
            weighted representation of input embeddings
        """
        if v is None:
            hidden = hidden.transpose(0, 1).transpose(1, 2)

        if hidden.shape[0] != english.shape[0]:
            hidden = hidden.transpose(0, 2)

        dotproduct = torch.bmm(english, hidden).squeeze(2)
        dotproduct = dotproduct / self.normaliser
        weights = self.softmax(dotproduct)
        if not validation:
            weights = F.dropout(weights, p=dropout_rate)
        self.last_weights = weights.data[0, :]
        if len(weights.shape) == 2:
            weights = weights.unsqueeze(1)
        if v is None:
            return torch.bmm(weights, english)
        else:
            return torch.bmm(weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()
        # Save weights for visualisation
        self.last_weights1 = None
        self.last_weights2 = None
        self.last_weights3 = None
        self.last_weights4 = None
        self.last_weights5 = None
        self.last_weights6 = None

        hidden_small = int(hidden_size/3)
        self.w_q1 = nn.Linear(hidden_size, hidden_small)
        self.w_q2 = nn.Linear(hidden_size, hidden_small)
        self.w_q3 = nn.Linear(hidden_size, hidden_small)
        self.w_k1 = nn.Linear(hidden_size, hidden_small)
        self.w_k2 = nn.Linear(hidden_size, hidden_small)
        self.w_k3 = nn.Linear(hidden_size, hidden_small)
        self.w_v1 = nn.Linear(hidden_size, hidden_small)
        self.w_v2 = nn.Linear(hidden_size, hidden_small)
        self.w_v3 = nn.Linear(hidden_size, hidden_small)

        self.w_o = nn.Linear(hidden_small*3, hidden_size)
        self.dotattention = ScaledDotAttention(hidden_small)

    def forward(self, english, hidden, validation, dropout_rate):
        """Forward pass of Multiheadattention mechanism with three heads.

        Args:
            english (Variable FloatTensor): embeddings
            hidden (Variable FloatTensor): last hidden state of decoder
            validation (bool): whether to apply dropout
            dropout_rate (float): how much dropout to apply

        Returns:
            weighted representation of input embeddings
        """
        hidden = hidden.transpose(0, 1)

        # Head 1
        k1 = self.w_k1(english)
        v1 = self.w_v1(english)
        q1 = self.w_q1(hidden).transpose(1, 2)
        head1 = self.dotattention(k1, q1, validation, dropout_rate, v1)
        self.last_weights1 = self.dotattention.last_weights

        # Head 2
        k2 = self.w_k2(english)
        v2 = self.w_v2(english)
        q2 = self.w_q2(hidden).transpose(1, 2)
        head2 = self.dotattention(k2, q2, validation, dropout_rate, v2)
        self.last_weights2 = self.dotattention.last_weights

        # Head 3
        k3 = self.w_k3(english)
        v3 = self.w_v3(english)
        q3 = self.w_q3(hidden).transpose(1, 2)
        head3 = self.dotattention(k3, q3, validation, dropout_rate, v3)
        self.last_weights3 = self.dotattention.last_weights

        heads = torch.cat((head1, head2, head3), dim=2)
        return self.w_o(heads)