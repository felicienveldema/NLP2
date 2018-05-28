import argparse
import logging
import pickle
import torch
import os
import pyter
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

from Encoder import Encoder, TransformerEncoder
from Decoder import Decoder

# Beginning and end of sentence tokens
BOS = "<s>"
EOS = "</s>"


def greedy(encoder, decoder, english, positions, w2i, i2w, max_length,
           enable_cuda):
    """Generate a translation one by one, by greedily selecting every new
    new word.

    Args:
        encoder: custom Encoder object
        decoder: custom Decoder object
        english: list of English words
        positions: list of word positions
        w2i (dict): mapping words to indices
        i2w (dict): mapping indices to words
        max_length (int): maximum generation length
        enable_cuda (bool): whether to enable CUDA

    Returns:
        translation: list of words
        all_weights: list of lists containing attention weights
    """

    # Initialise the variables and the encoding
    english, positions, french_token = prepare_variables(
        english, positions, w2i, enable_cuda
    )
    translation = []
    h_dec, english = encoder.eval(english, positions)
    all_weights = []

    # Sample words one by one
    for j in range(1, max_length + 1):
        vocab_probs, h_dec, weights = decoder.eval(french_token, english, h_dec)
        all_weights.append(list(weights))
        _, french_token = torch.topk(vocab_probs, 1)
        french_token = french_token[:, 0]
        translation.append(i2w[french_token.data[0]])

        # If the </s> token is found, end generating
        if translation[-1] == EOS:
            break

    return translation, all_weights


def beam(encoder, decoder,english, positions, w2i, i2w, max_length, enable_cuda,
         beam_size=5):
    """Generate a translation according to beam search, with k=5.

    Args:
        encoder: custom Encoder object
        decoder: custom Decoder object
        english: list of English words
        positions: list of word positions
        w2i (dict): mapping words to indices
        i2w (dict): mapping indices to words
        max_length (int): maximum generation length
        enable_cuda (bool): whether to enable CUDA
        beam_size (int): beam size, default is 5

    Returns:
        translation: list of words
    """
    english, positions, french_token = prepare_variables(
        english, positions, w2i, enable_cuda
    )
    h_dec, english = encoder.eval(english, positions)

    new_hypotheses = [([BOS], h_dec, french_token, 0)]
    final = []

    for j in range(1, max_length + 1):
        hypotheses = new_hypotheses
        new_hypotheses = []
        for hypothesis in hypotheses:
            # Expand with K+1 new words, because at best K new candidates will
            # be kept for the next round, and the EOS may pop up
            hyp_extended, final_extend = extend_beam(
                hypothesis, decoder, i2w, english, h_dec, beam_size, enable_cuda
            )
            new_hypotheses.extend(hyp_extended)
            final.extend(final_extend)
        new_hypotheses.sort(key=lambda x: x[3], reverse=True)
        new_hypotheses = new_hypotheses[:beam_size]

    # Normalize for the length
    hypotheses = Counter()
    for hypothesis, _, _, prob in new_hypotheses + final:
        prob = prob / len(hypothesis)
        hypotheses[tuple(hypothesis)] = prob

    translation = list(hypotheses.most_common(1)[0][0])
    return translation, None


def clean(sequence):
    # Remove BOS en EOS tokens because they should not be taken into account in BLEU
    if BOS in sequence: sequence.remove(BOS)
    if EOS in sequence: sequence.remove(EOS)
    return sequence

def test(corpus, test_pairs, max_length, enable_cuda, epoch, transformer=False):

    scores_bleu = []
    scores_ter = []
    chencherry = SmoothingFunction()
    greedy_ref = open("greedy.ref", 'w', encoding='utf8')
    greedy_hyp = open("greedy.hyp", 'w', encoding='utf8')
    for i, (english, french) in tqdm(enumerate(list(zip(test_pairs[0], test_pairs[1])))):
        positions = corpus.word_positions(english)
        indices = corpus.to_indices(english)
        translation, attention  = greedy(
            encoder, decoder, indices, positions, corpus.dict_f.word2index,
            corpus.dict_f.index2word, max_length, enable_cuda
        )

        if i==35 and transformer:
            data = [go.Heatmap(z=attention, x=english, y=translation, colorscale='Viridis')]
            layout = go.Layout(width=800, height=600)
            fig = go.Figure(data=data, layout=layout)
            py.image.save_as(fig, filename='weights_{}.png'.format(epoch))
            attention1 = encoder.layer1.attention.last_weights1
            attention2 = encoder.layer1.attention.last_weights2
            attention3 = encoder.layer1.attention.last_weights3
            with open("weights_{}.txt".format(epoch), 'w') as f:
                f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention1]))
                f.write("\n")
                f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention2]))
                f.write("\n")
                f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention3]))
                f.write("\n")
                f.write("\t".join(english))
                f.write("\t".join(translation))
        elif i == 35:
            data = [go.Heatmap(z=attention, x=english, y=translation, colorscale='Viridis')]
            layout = go.Layout(width=800, height=600)
            fig = go.Figure(data=data, layout=layout)
            py.image.save_as(fig, filename='weights_{}.png'.format(epoch))
            with open("weights_{}.txt".format(epoch), 'w') as f:
                f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention]))
                f.write("\n")
                f.write("\t".join(english))
                f.write("\t".join(translation))

        french = clean(corpus.bpe_to_sentence(french))
        translation = clean(corpus.bpe_to_sentence(translation))
        scores_bleu.append(sentence_bleu([french], translation, smoothing_function=chencherry.method1))
        scores_ter.append(pyter.ter(translation, french))
        greedy_ref.write(" ".join(french) + "\n")
        greedy_hyp.write(" ".join(translation) + "\n")
    greedy_ref.close()
    greedy_hyp.close()
    score_bleu = sum(scores_bleu) / len(scores_bleu)
    score_ter = sum(scores_ter) / len(scores_ter)
    logging.info("Greedy, BLEU: {}, TER: {}, METEOR".format(score_bleu, score_ter))
    scores_bleu = []
    scores_ter = []

    beam_ref = open("beam.ref", 'w', encoding='utf8')
    beam_hyp = open("beam.hyp", 'w', encoding='utf8')
    lengths = []
    for english, french in tqdm(list(zip(test_pairs[0], test_pairs[1]))):
        positions = corpus.word_positions(english)
        indices = corpus.to_indices(english)
        translation, attention  = beam(
            encoder, decoder, indices, positions, corpus.dict_f.word2index,
            corpus.dict_f.index2word, max_length, enable_cuda
        )

        if i == 35:
            # Attention visualization
            data = [go.Heatmap(z=attention, x=english, y=translation, colorscale='Viridis')]
            layout = go.Layout(width=800, height=600)
            fig = go.Figure(data=data, layout=layout)
            py.image.save_as(fig, filename='weights_{}.png'.format(epoch))
            with open("weights_{}.txt".format(epoch), 'w') as f:
                f.write("\n".join(["\t".join([str(num) for num in line]) for line in attention]))
                f.write("\n")
                f.write("\t".join(english))
                f.write("\t".join(translation))

        french = clean(corpus.bpe_to_sentence(french))
        translation = clean(corpus.bpe_to_sentence(translation))
        scores_bleu.append(sentence_bleu([french], translation, smoothing_function=chencherry.method1))
        scores_ter.append(pyter.ter(translation, french))
        beam_ref.write(" ".join(french) + "\n")
        beam_hyp.write(" ".join(translation) + "\n")
        lengths.append(len(french))
    beam_ref.close()
    beam_hyp.close()

    score_bleu = sum(scores_bleu) / len(scores_bleu)
    score_ter = sum(scores_ter) / len(scores_ter)
    logging.info("Beam, BLEU: {}, TER: {}, METEOR".format(score_bleu, score_ter))
    with open("lengths.txt", 'w') as f:
        f.write("\n".join([ str(l) for l in lengths ]))
    with open("bleu.txt", 'w') as f:
        f.write("\n".join([ str(l) for l in scores_bleu ]))
    with open("ter.txt", 'w') as f:
        f.write("\n".join([ str(l) for l in scores_ter ]))


def extend_beam(hypothesis, decoder, i2w, english, h_dec, beam_size,
                enable_cuda):
    """Expand one hypothesis from the beam with k + 1 new hypotheses.

    Args:
        hypothesis (list): words generated so far
        decoder: custom Decoder object
        i2w (dict): mapping indices to words
        english (Variable): containing English word indices
        h_dec (Variable): former hidden state of decoder
        beam_size (int): size of the beam
        enable_cuda (bool): whether cuda is available

    Returns:
        new_hypotheses: list of extended hypotheses
        final: list of finished hypotheses found
    """
    # Draw the top k + 1 words from the decoder
    translation, h_dec, french_token, prob = hypothesis
    vocab_probs, h_dec, _ = decoder.eval(french_token, english, h_dec)
    top_probs, french_token = torch.topk(vocab_probs, beam_size + 1)
    top_probs = list(top_probs.data[0])
    french_token = list(french_token.data[0])

    new_hypotheses = []
    final = []
    for word_prob, token in zip(top_probs, french_token):
        word = i2w[token]

        # If the EOS token is in the top, add the hypothesis to the final list
        if EOS == word:
            final.append((translation + [word], h_dec, None, prob + word_prob))
        else:
            french = torch.autograd.Variable(torch.LongTensor([token]))
            if enable_cuda: french = french.cuda()
            new_hypotheses.append((translation + [word], h_dec, french,
                                   prob + word_prob))
    return new_hypotheses, final


def prepare_variables(english, positions, w2i, enable_cuda):
    """Turn words and positions into Variables containing Longtensors.

    Args:
        english (list): list of english words
        positions (list): list of positions of english words
        w2i (dict): mapping words to indices
        enable_cuda (bool): whether cuda is available

    Returns;
        english: now a Variable containing a Longtensor
        positions: now a Variable containing a Longtensor
        french_token: Variable with the first french token "<s>"
    """
    english = torch.autograd.Variable(torch.LongTensor([english]))
    positions = torch.autograd.Variable(torch.LongTensor([positions]))
    french_token = torch.autograd.Variable(torch.LongTensor([w2i[BOS]]))

    if enable_cuda:
        english = english.cuda()
        positions = positions.cuda()
        french_token = french_token.cuda()
    return english, positions, french_token

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english', type=str,   default='test/test_2017_flickr.en')
    p.add_argument('--french',  type=str,   default='test/test_2017_flickr.fr')
    p.add_argument('--encoder', type=str,   default='encoder_type=gru.pt')
    p.add_argument('--decoder', type=str,   default='decoder_type=gru.pt')
    p.add_argument('--corpus',  type=str,   default='corpus.pickle')
    p.add_argument('--max_length',    	type=int,   default=74)
    p.add_argument('--enable_cuda',   	action='store_true')
    p.add_argument('--transformer', 	action='store_true')

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

    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)
    corpus = pickle.load(open(args.corpus, 'rb'))
    corpus.dict_e.word2index = { k: v for k, v in corpus.dict_e.word2index }
    corpus.dict_e.index2word = { k: v for k, v in corpus.dict_e.index2word }
    corpus.dict_f.word2index = { k: v for k, v in corpus.dict_f.word2index }
    corpus.dict_f.index2word = { k: v for k, v in corpus.dict_f.index2word }
    test_pairs = corpus.load_data(args.english, args.french)
    test(corpus, test_pairs, args.max_length, enable_cuda, 0, args.transformer)