import argparse
import logging
import pickle
import torch
import os
import torch.nn as nn
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch import optim
from collections import defaultdict, Counter
from random import shuffle, random
from data import Corpus
from bpe import learn_bpe, BPE

from Encoder import Encoder
from Decoder import Decoder


def train(corpus, encoder, decoder, lr, epochs, batch_size, enable_cuda, ratio):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) +
                                 list(decoder.parameters()), lr=lr)
    losses = []
    for i in range(epochs):
        epoch_loss = 0
        for english, english_pos, french, french_pos in tqdm(corpus.batches):
            optimizer.zero_grad()
            use_teacher_forcing = True if random() < ratio else False

            # First run the encoder, it encodes the English sentence
            h_enc = encoder.init_hidden(batch_size, enable_cuda)
            h_dec = encoder(english, english_pos, h_enc)

            # Now go through the decoder step by step and use teacher forcing
            # for a ratio of the batches
            french_token = french[:, 0]
            loss = 0
            for j in range(1, french.shape[1]):
                vocab_probs, h_dec = decoder(french_token, h_dec)
                loss += criterion(vocab_probs, french[:, j])
                if use_teacher_forcing:
                    french_token = french[:, j]
                else:
                    _, french_token = torch.topk(vocab_probs, 1)
                    french_token = french_token[:, 0]

            # Now update the weights
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0] / french.shape[1]
        epoch_loss = epoch_loss / len(corpus.batches)
        logging.info("Loss per token: {}".format(epoch_loss))
        losses.append(epoch_loss)
    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english',     type=str,   default='training/train.en')
    p.add_argument('--french',      type=str,   default='training/train.fr')
    p.add_argument('--enc_type',    type=str,   default='avg')
    p.add_argument('--dec_type',    type=str,   default='rnn')
    p.add_argument('--lr',          type=float, default=0.001)
    p.add_argument('--tf_ratio',    type=float, default=0.75)
    p.add_argument('--batch_size',  type=int,   default=1)
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--dim',         type=int,   default=100)
    p.add_argument('--num_symbols', type=int,   default=10000)
    p.add_argument('--min_count',   type=int,   default=1)
    p.add_argument('--max_length',  type=int,   default=100)
    p.add_argument('--lower',       action='store_true')
    p.add_argument('--enable_cuda', action='store_true')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Check whether GPU is present
    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        #torch.cuda.set_device(0)
        logging.info("CUDA is enabled")
    else:
        enable_cuda = False
        logging.info("CUDA is disabled")

    # Prepare corpus, encoder and decoder
    corpus = Corpus(args.english, args.french, args.batch_size,args.num_symbols,
                    args.min_count, args.lower, args.enable_cuda)
    encoder = Encoder(args.dim, corpus.vocab_size_e, corpus.max_pos,
                      args.enc_type)
    print(corpus.dict_f.word2index["<s>"])
    print(corpus.dict_f.word2index["</s>"])
    eos = corpus.dict_f.word2index["</s>"]
    decoder = Decoder(args.dim, corpus.vocab_size_f, eos, args.dec_type)

    # Train
    losses = train(corpus, encoder, decoder, args.lr, args.epochs,
                   args.batch_size, enable_cuda, args.tf_ratio)

    # Plot losses and save figure
    plt.figure(figsize=(15, 10))
    plt.plot([i for i in range(len(losses))], losses)
    plt.scatter([i for i in range(len(losses))], losses)
    plt.savefig("loss_enc={}_dec={}.png".format(args.enc_type, args.dec_type))
