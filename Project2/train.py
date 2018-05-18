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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english', type=str, default='training/train.en',
                   help='path to Fnglish data.')
    p.add_argument('--french', type=str, default='training/train.fr',
                   help='path to French data.')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--batch_size', type=int, default=64, help='batch size')
    p.add_argument('--enable_cuda', action='store_true', help='use CUDA')
    p.add_argument('--epochs', type=int, default=10, help='#epochs')
    p.add_argument('--dim', default=50, type=int)
    p.add_argument('--nr_sents', default=-1, type=int)
    p.add_argument('--unique_words', default=10000, type=int)
    p.add_argument('--min_count', default=0, type=int)
    p.add_argument('--lower', action='store_true')
    
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
    corpus = Corpus(args.english, args.french, args.batch_size, args.nr_sents,
                    args.unique_words, args.min_count, args.lower,
                    args.enable_cuda)
    logging.info("Loaded data.")
    
    