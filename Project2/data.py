import nltk
import sys
import logging
import random
import torch
import pickle
import os
import numpy as np

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch import LongTensor
from torch.autograd import Variable
from collections import defaultdict, Counter
from bpe import learn_bpe, BPE
from copy import deepcopy


class Dictionary(object):
    """Object that creates and keeps word2index and index2word dicts."""

    def __init__(self):
        """Initialize word - index mappings."""
        self.word2index = defaultdict(lambda: len(self.word2index))
        self.index2word = dict()
        self.counts = Counter()

    def add_word(self, word):
        """Add one new word to your dicitonary.

        Args:
            word (str): word to add to dictionary
        """
        index = self.word2index[word]
        self.index2word[index] = word
        self.counts[word] += 1
        return index

    def add_text(self, text):
        """Add a list of words to your dictionary.

        Args:
            text (list of strings): text to add to dictionary
        """
        for word in text:
            self.add_word(word)

    def to_unk(self):
        """From now on your dictionaries default to UNK for unknown words."""
        unk = self.add_word("UNK")
        self.word2index = defaultdict(lambda: unk, self.word2index)


class Corpus(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, pathl1, pathl2, batch_size, nr_docs, nr_unique_words,
                 min_count, lower, enable_cuda=False):
        """Initialize pairs of words and associations.

        Args:
            pathl1 (str): the path to the L1 data
            pathl2 (str): the path to the L2 data
            batch_size (int): int indicating the desired size for batches
            nr_docs (int): how many sentences should be used from the corpus
            enable_cuda (bool): whether to cuda the batches
        """
        self.batch_size = batch_size
        self.dict_e = Dictionary()
        self.dict_f = Dictionary()
        self.lines_e = []
        self.lines_f = []
        self.enable_cuda = enable_cuda

        # Read in the corpus
        with open(pathl1, 'r', encoding='utf8') as f_eng, open(pathl2, 'r', encoding='utf8') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                line_e = self.prepare(line_e, lower)
                line_f = self.prepare(line_f, lower)
                self.dict_e.add_text(line_e)
                self.dict_f.add_text(line_f)
                self.lines_e.append(line_e)
                self.lines_f.append(line_f)
                if len(self.lines_e) == nr_docs and nr_docs != -1:
                    break

        learn_bpe(self.dict_e.counts, open("learned_bpe_e.txt", 'w', encoding='utf8'), 10000, min_frequency=40)
        learn_bpe(self.dict_f.counts, open("learned_bpe_f.txt", 'w', encoding='utf8'), 10000, min_frequency=40)
        bpe_f = BPE(open("learned_bpe_f.txt", 'r', encoding='utf8'), vocab=deepcopy(self.dict_f.counts))
        bpe_e = BPE(open("learned_bpe_e.txt", 'r', encoding='utf8'), vocab=deepcopy(self.dict_e.counts))

        # Read in the corpus
        self.dict_e = Dictionary()
        self.dict_f = Dictionary()
        with open(pathl1, 'r', encoding='utf8') as f_eng, open(pathl2, 'r', encoding='utf8') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                if lower:
                    line_e = line_e.lower()
                    line_f = line_f.lower()

                line_e = bpe_e.process_line(line_e)
                line_f = bpe_f.process_line(line_f)
                self.dict_e.add_text(line_e)
                self.dict_f.add_text(line_f)

        # Update dictionaries and map to UNK
        self.vocab_size_e = len(self.dict_e.word2index)
        self.vocab_size_f = len(self.dict_f.word2index)

        # Create batches
        self.batches = self.get_batches(enable_cuda)
        logging.info("Created Corpus.")
                     
    def get_batches(self, enable_cuda):
        """Create batches from data in class.

        Args:
            enable_cuda (bool): cuda batches or not

        Returns:
            list of batches
        """
        # Sort lines by the length of the English sentences
        sorted_lengths = [[len(x), len(y), x, y]
                               for x,y in zip(self.lines_e, self.lines_f)]
        sorted_lengths.sort()
        eng_sents = [x[2] for x in sorted_lengths]
        fre_sents = [x[3] for x in sorted_lengths]

        batches = []
        
        # Go through data in steps of batch size
        for i in range(0, len(eng_sents) - self.batch_size, self.batch_size):
            max_french = max([x[1] for x in sorted_lengths[i:i+self.batch_size]])
            max_english = max([x[0] for x in sorted_lengths[i:i+self.batch_size]])
            batch_french = LongTensor(self.batch_size, max_french)
            batch_english = LongTensor(self.batch_size, max_english)

            batch_lines_e = eng_sents[i:i+self.batch_size]
            batch_lines_f = fre_sents[i:i+self.batch_size]
            for j, (eline, fline) in enumerate(zip(batch_lines_e, batch_lines_f)):
                # Map words to indices and pad with EOS tag
                fline = self.pad_list(
                    fline, False, max_french, pad=self.dict_f.word2index['</s>']
                )
                eline = self.pad_list(
                    eline, True, max_english, pad=self.dict_e.word2index['</s>']
                )
                
                batch_french[j, :] = LongTensor(fline)
                batch_english[j,:] = LongTensor(eline)

            if enable_cuda:
                batches.append((Variable(batch_english).cuda(), Variable(batch_french).cuda()))
            else:
                batches.append((Variable(batch_english), Variable(batch_french)))
        random.shuffle(batches)
        return batches

    def pad_list(self, line, english, length, pad=0):
        """Pads list to a certain length
        
        Args:
            input (list): list to pad
            length (int): length to pad to
            pad (object): object to pad with
        """
        line = self.to_indices(line, english)
        return line + [pad] * max(0,length - len(line))
        
    def to_indices(self, sequence, english=True):
        """Represent a history of words as a list of indices.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        if english:
            return [self.dict_e.word2index[w] for w in sequence]
        else:
            return [self.dict_f.word2index[w] for w in sequence]

    def prepare(self, sequence, lower):
        """Add start and end tags. Add words to the dictionary.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        if lower: sequence = sequence.lower()
        return ['<s>'] + word_tokenize(sequence) + ['</s>']
