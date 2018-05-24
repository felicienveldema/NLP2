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

import plotly.plotly as py 
import plotly.graph_objs as go
import plotly.offline as offline
import plotly
plotly.tools.set_credentials_file(username='vdankers', api_key='iqYAxJNr16lmrFjgys4l')

from Encoder import Encoder
from Decoder import Decoder
from test import greedy, beam

# Beginning and end of sentence tokens
BOS = "<s>"
EOS = "</s>"

def clean(sequence):
    # Remove BOS en EOS tokens because they should not be taken into account in BLEU
    if BOS in sequence: sequence.remove(BOS)
    if EOS in sequence: sequence.remove(EOS)
    return sequence

def validate(corpus, valid, max_length, enable_cuda, epoch):

    scores = []
    chencherry = SmoothingFunction()
    for english, french in tqdm(list(zip(valid[0], valid[1]))):
        positions = corpus.word_positions(english)
        indices = corpus.to_indices(english)
        translation, attention  = greedy(
            encoder, decoder, indices, positions, corpus.dict_f.word2index,
            corpus.dict_f.index2word, max_length, enable_cuda
        )

        # if i == 35:
        #     #data = [go.Heatmap(z=attention, x=english, y=translation, colorscale='Viridis')]
        #     #layout = go.Layout(width=800, height=600)
        #     #fig = go.Figure(data=data, layout=layout)
        #     #py.image.save_as(fig, filename='weights_{}.png'.format(epoch))
        #     with open("weights_{}.txt".format(epoch), 'w') as f:
        #         f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention]))
        #         f.write("\n")
        #         f.write("\t".join(english))
        #         f.write("\t".join(translation))

        french = clean(corpus.bpe_to_sentence(french))
        translation = clean(corpus.bpe_to_sentence(translation))
        scores.append(sentence_bleu([french], translation, smoothing_function=chencherry.method1))


    score = sum(scores) / len(scores)
    print("Greedy average BLEU score: {}".format(score))

    return score


def train(corpus, valid, encoder, decoder, lr, epochs, batch_size, enable_cuda,
          ratio, max_length):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) +
                                 list(decoder.parameters()), lr=lr)
    losses = []
    bleus = []
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

        bleus.append(validate(corpus, valid, max_length, enable_cuda, i))
        # Dropout annealing
        encoder.dropout_rate = max([0, 1 - i / 20]) * encoder.dropout_rate_0
        decoder.dropout_rate = max([0, 1 - i / 20]) * decoder.dropout_rate_0
        print(encoder.dropout_rate)
        print(decoder.dropout_rate)


        epoch_loss = epoch_loss / len(corpus.batches)
        logging.info("Loss per token: {}".format(epoch_loss))
        losses.append(epoch_loss)

    return losses, bleus


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
    p.add_argument('--max_length',    type=int,   default=74)
    p.add_argument('--lower',         action='store_true')
    p.add_argument('--enable_cuda',   action='store_true')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Check whether GPU is present
    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        torch.cuda.set_device(1)
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
    if enable_cuda:
        encoder.cuda()
        decoder.cuda()

    # Train
    losses, bleus = train(corpus, valid, encoder, decoder, args.lr, args.epochs,
                          args.batch_size, enable_cuda, args.tf_ratio,
                          args.max_length)

    # Plot losses and save figure
    plt.figure(figsize=(15, 10))
    plt.plot([i for i in range(len(losses))], losses)
    plt.scatter([i for i in range(len(losses))], losses)
    plt.savefig("loss_enc={}_dec={}_att={}.png".format(args.enc_type, args.dec_type, args.attention))

    plt.figure(figsize=(15, 10))
    plt.plot([i for i in range(len(bleus))], bleus)
    plt.scatter([i for i in range(len(bleus))], bleus)
    plt.savefig("bleu_enc={}_dec={}_att={}.png".format(args.enc_type, args.dec_type, args.attention))

    torch.save(encoder, "encoder_type={}.pt".format(args.enc_type))
    torch.save(decoder, "decoder_type={}.pt".format(args.dec_type))
    pickle.dump(corpus, open("corpus.pickle", 'wb'))