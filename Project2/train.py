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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from Encoder import Encoder
from Decoder import Decoder


def greedy(english, positions, w2i, i2w, max_length, enable_cuda):

    english = torch.autograd.Variable(torch.LongTensor([english]))
    positions = torch.autograd.Variable(torch.LongTensor([positions]))
    translation = []
    h_enc = encoder.init_hidden(1, enable_cuda)
    h_dec, english = encoder(english, positions, h_enc)
    french_token = torch.autograd.Variable(torch.LongTensor([w2i["<s>"]]))

    for j in range(1, max_length):
        vocab_probs, h_dec = decoder(french_token, english, h_dec)
        _, french_token = torch.topk(vocab_probs, 1)
        french_token = french_token[:, 0]
        translation.append(i2w[french_token.data[0]])
        if translation[-1] == "</s>":
            break
    return translation


def validate(corpus, valid, max_length, enable_cuda):
    scores = []
    chencherry = SmoothingFunction()
    for english, french in zip(valid[0], valid[1]):
        positions = corpus.word_positions(english)
        indices = corpus.to_indices(english)
        translation  = greedy(indices, positions, corpus.dict_f.word2index,
                              corpus.dict_f.index2word, max_length, enable_cuda)
        french = corpus.bpe_to_sentence(french)
        translation = corpus.bpe_to_sentence(translation)
        scores.append(sentence_bleu([french], translation, smoothing_function=chencherry.method1))
    print("Average BLEU score: {}".format(sum(scores) / len(scores)))


def train(corpus, valid, encoder, decoder, lr, epochs, batch_size, enable_cuda,
          ratio, max_length):
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
            h_dec, english = encoder(english, english_pos, h_enc)

            # Now go through the decoder step by step and use teacher forcing
            # for a ratio of the batches
            french_token = french[:, 0]
            loss = 0
            for j in range(1, french.shape[1]):
                vocab_probs, h_dec = decoder(french_token, english, h_dec)
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

        validate(corpus, valid, max_length, enable_cuda)

        epoch_loss = epoch_loss / len(corpus.batches)
        logging.info("Loss per token: {}".format(epoch_loss))
        losses.append(epoch_loss)

    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english_train', type=str,   default='training/train.en')
    p.add_argument('--french_train',  type=str,   default='training/train.fr')
    p.add_argument('--english_valid', type=str,   default='val/val.en')
    p.add_argument('--french_valid',  type=str,   default='val/val.fr')
    p.add_argument('--enc_type',      type=str,   default='avg')
    p.add_argument('--dec_type',      type=str,   default='rnn')
    p.add_argument('--attention',     type=str,   default='dot')
    p.add_argument('--lr',            type=float, default=0.001)
    p.add_argument('--tf_ratio',      type=float, default=0.75)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--epochs',        type=int,   default=10)
    p.add_argument('--dim',           type=int,   default=100)
    p.add_argument('--num_symbols',   type=int,   default=10000)
    p.add_argument('--min_count',     type=int,   default=1)
    p.add_argument('--max_length',    type=int,   default=50)
    p.add_argument('--lower',         action='store_true')
    p.add_argument('--enable_cuda',   action='store_true')

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
    corpus = Corpus(args.english_train, args.french_train, args.batch_size,
                    args.num_symbols, args.min_count, args.lower,
                    args.enable_cuda)
    encoder = Encoder(args.dim, corpus.vocab_size_e, corpus.max_pos,
                      args.enc_type)
    valid = corpus.load_data(args.english_valid, args.french_valid)

    eos = corpus.dict_f.word2index["</s>"]
    decoder = Decoder(args.dim, corpus.vocab_size_f, eos, 
                      corpus.longest_english, args.dec_type, args.attention)

    # Train
    losses = train(corpus, valid, encoder, decoder, args.lr, args.epochs,
                   args.batch_size, enable_cuda, args.tf_ratio, args.max_length)

    # Plot losses and save figure
    plt.figure(figsize=(15, 10))
    plt.plot([i for i in range(len(losses))], losses)
    plt.scatter([i for i in range(len(losses))], losses)
    plt.savefig("loss_enc={}_dec={}_att={}.png".format(args.enc_type, args.dec_type, args.attention))

    torch.save(encoder, "encoder_type={}.pt".format(args.enc_type))
    torch.save(decoder, "decoder_type={}.pt".format(args.dec_type))
    pickle.dump(corpus, open("corpus.pickle", 'wb'))