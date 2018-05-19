import argparse
import logging
import pickle
import torch
import os
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from collections import defaultdict, Counter
from random import shuffle
from data import Corpus
from bpe import learn_bpe, BPE

from Encoder import Encoder
from Decoder import Decoder



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english', type=str, default='training/train.en')
    p.add_argument('--french', type=str, default='training/train.fr')
    p.add_argument('--lr',          type=float, default=0.01)
    p.add_argument('--batch_size',  type=int,   default=1)
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--dim',         type=int,   default=100)
    p.add_argument('--num_symbols', type=int,   default=10000)
    p.add_argument('--min_count',   type=int,   default=1)
    p.add_argument('--max_length',  type=int,   default=100)
    p.add_argument('--lower', action='store_true')
    p.add_argument('--enable_cuda', action='store_true', help='use CUDA')

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

    # Prepare corpus + dictionaries, create training batches
    corpus = Corpus(args.english, args.french, args.batch_size,args.num_symbols,
                    args.min_count, args.lower, args.enable_cuda)
    encoder = Encoder(args.dim, args.dim)
    print(corpus.dict_f.word2index["</s>"])
    decoder = Decoder(args.dim, corpus.vocab_size_f, corpus.dict_f.word2index["</s>"])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    for i in range(args.epochs):
        for english, english_pos, french, french_pos in tqdm(corpus.batches):
            h_enc = encoder.init_hidden(args.batch_size, enable_cuda)
            h_dec = decoder.init_hidden(args.batch_size, enable_cuda)
            _, encoder_output = encoder(english, h_enc)

            loss = 0
            for j in range(french.shape[1]):
                vocab_probs, h_dec = decoder(encoder_output, h_dec)
                print(french[:, j].shape)
                print(vocab_probs.shape)
                loss += criterion(french[:, j].data, vocab_probs)
            print(values, indices)
            exit()