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


class TransformerSublayer(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(TransformerSublayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feedforward1 = nn.Linear(hidden_size, hidden_size * 4)
        self.feedforward2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, english, dropout_rate, validation=False):
        """Forward pass of Transformer layer containing two sublayers:
            1. Self-attention mechanism
            2. Feedforward neural network

        Args:
            english (Variable FloatTensor): encoding of English sentence
            dropout_rate (float): current dropout rate
            validation (bool): if validating, don't apply dropout

        Returns:
            attention: attention applied to input embeddings
            feedforward output
        """
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

        # Layers with sublayers of self-attention and feedforward
        self.layer1 = TransformerSublayer(hidden_size, input_size)
        self.layer2 = TransformerSublayer(hidden_size, input_size)

    def forward(self, english, positions, validation=False):
        """Forward pass for the Transformer encoder.

        Args:
            english (Variable LongTensor): indices of english words
            positions (Variable LongTensor): indices of word positions
            validation (bool): whether you are validating (don't use dropout)

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        english_embed = self.embeddings(english)
        pos_embed = self.position_embeddings(positions)

        if not validation:
            english_embed = F.dropout(english_embed, p=self.dropout_rate)
            pos_embed = F.dropout(pos_embed, p=self.dropout_rate)

        english = torch.add(english_embed, pos_embed)
        hidden, output1 = self.layer1(english, self.dropout_rate, validation)
        hidden, output2 = self.layer2(output1, self.dropout_rate, validation)
        return torch.mean(hidden, dim=1).unsqueeze(0), output2

    def eval(self, english, positions):
        """Evaluation pass for the encoder: don't apply dropout.

        Args:
            english (Variable LongTensor): indices of english words
            positions (Variable LongTensor): indices of word positions

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        return self.forward(english, positions, True)


class Encoder(nn.Module):
    def __init__(self, hidden_size, input_size, max_pos, type, enable_cuda):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.type = type
        self.enable_cuda = enable_cuda

        # Encoder is gru or avg
        if self.type.lower() == 'gru':
            self.network = nn.GRU(hidden_size, hidden_size, batch_first=True,
                                  bidirectional=True)

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.position_embeddings = nn.Embedding(100, hidden_size)

    def forward(self, english, positions, validation=False):
        """Forward pass for the encoder.

        Args:
            english (Variable LongTensor): indices of english words
            positions (Variable LongTensor): indices of word positions
            validation (bool): whether you are validating (don't use dropout)

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        hidden = self.init_hidden(english.shape[0], self.enable_cuda)
        english_embed = self.embeddings(english)
        pos_embed = self.position_embeddings(positions)

        # Don't apply dropout if validating
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
        """Evaluation pass for the encoder: don't apply dropout.

        Args:
            english (Variable LongTensor): indices of english words
            positions (Variable LongTensor): indices of word positions

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        return self.forward(english, positions, True)

    def init_hidden(self, batch_size, enable_cuda):
        """Initialize the first hidden state randomly.

        Args:
            batch_size (int)
            enable_cuda (bool): whether a GPU is available

        Returns:
            Variable FloatTensor
        """
        if enable_cuda:
            return Variable(torch.randn(2, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(2, batch_size, self.hidden_size))