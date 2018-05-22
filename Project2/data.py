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

    def __init__(self, pathl1, pathl2, batch_size, num_symbols,
                 min_count, lower, enable_cuda=False):
        """Initialize pairs of words and associations.

        Args:
            pathl1 (str): the path to the L1 data
            pathl2 (str): the path to the L2 data
            batch_size (int): int indicating the desired size for batches
            num_symbols (int): number of symbols for BPE
            min_count (int): word must occur at least min count times
            lower (bool): if true, sentences are lowercased
            enable_cuda (bool): whether to cuda the batches
        """
        self.batch_size = batch_size
        self.dict_e = Dictionary()
        self.dict_f = Dictionary()
        self.lines_e = []
        self.lines_f = []
        self.enable_cuda = enable_cuda
        self.max_pos = 0
        self.lower = lower

        # Read in the corpus
        with open(pathl1, 'r', encoding='utf8') as f_eng, open(pathl2, 'r', encoding='utf8') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                line_e = self.prepare(line_e, lower)
                line_f = self.prepare(line_f, lower)
                self.dict_e.add_text(line_e)
                self.dict_f.add_text(line_f)

        # Learn BPEs
        out = open("learned_bpe_e.txt", 'w', encoding='utf8')
        learn_bpe(self.dict_e.counts, out, num_symbols, min_frequency=min_count)
        out = open("learned_bpe_e.txt", 'r', encoding='utf8')
        self.bpe_e = BPE(out, vocab=deepcopy(self.dict_e.counts))

        out = open("learned_bpe_f.txt", 'w', encoding='utf8')
        learn_bpe(self.dict_f.counts, out, num_symbols, min_frequency=min_count)
        out = open("learned_bpe_f.txt", 'r', encoding='utf8')
        self.bpe_f = BPE(out, vocab=deepcopy(self.dict_f.counts))

        # Read in the corpus again, now with BPE adapted vocabulary
        self.dict_e = Dictionary()
        self.dict_f = Dictionary()

        lines_e, lines_f = self.load_data(pathl1, pathl2)
        for line_e, line_f in zip(lines_e, lines_f):
            self.dict_e.add_text(line_e)
            self.dict_f.add_text(line_f)
            self.lines_e.append(line_e)
            self.lines_f.append(line_f)

        self.longest_english = max([len(e) for e in lines_e])
        #print(self.longest_english)
        #self.dict_e.to_unk()
        #self.dict_f.to_unk()
        self.vocab_size_e = len(self.dict_e.word2index)
        self.vocab_size_f = len(self.dict_f.word2index)

        # Create batches
        self.batches = self.get_batches(enable_cuda)
        logging.info("Created Corpus.")

    def load_data(self, pathl1, pathl2):
        lines_e = []
        lines_f = []
        with open(pathl1, 'r', encoding='utf8') as f_eng, open(pathl2, 'r', encoding='utf8') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                line_e = " ".join(self.prepare(line_e, self.lower))
                line_f = " ".join(self.prepare(line_f, self.lower))
                line_e = self.bpe_e.process_line(line_e).split()
                line_f = self.bpe_f.process_line(line_f).split()
                lines_e.append(line_e)
                lines_f.append(line_f)
        return lines_e, lines_f
                     
    def get_batches(self, enable_cuda):
        """Create batches from data in class.

        Args:
            enable_cuda (bool): cuda batches or not

        Returns:
            list of batches
        """
        # Sort lines by the length of the English sentences
        sorted_lengths = [[len(x), len(y), self.word_positions(x), self.word_positions(y), x, y]
                               for x,y in zip(self.lines_e, self.lines_f)]
        sorted_lengths.sort()
        
        batches = []
        
        # Go through data in steps of batch size
        for i in range(0, len(sorted_lengths) - self.batch_size, self.batch_size):
            max_french = max([x[1] for x in sorted_lengths[i:i+self.batch_size]])
            max_english = max([x[0] for x in sorted_lengths[i:i+self.batch_size]])
            batch_french = LongTensor(self.batch_size, max_french)
            batch_english = LongTensor(self.batch_size, max_english)
            batch_english_pos = LongTensor(self.batch_size, max_english)
            batch_french_pos = LongTensor(self.batch_size, max_french)

            for j, data in enumerate(sorted_lengths[i:i+self.batch_size]):
                # Map words to indices and pad with EOS tag
                fline = self.pad_list(
                    data[5], False, max_french, pad=self.dict_f.word2index['</s>']
                )
                eline = self.pad_list(
                    data[4], True, max_english, pad=self.dict_e.word2index['</s>']
                )

                batch_french[j, :] = LongTensor(fline)
                batch_english[j,:] = LongTensor(eline)

                e_pos = data[2] + [data[2][-1]]*(max_english - len(data[2]))
                f_pos = data[3] + [data[3][-1]]*(max_french - len(data[3]))
                batch_english_pos[j,:] = LongTensor(e_pos)
                batch_french_pos[j,:] = LongTensor(f_pos)
                
            batch_english = Variable(batch_english)
            batch_english_pos = Variable(batch_english_pos)
            batch_french = Variable(batch_french)
            batch_french_pos = Variable(batch_french_pos)

            if enable_cuda:
                batch_english = batch_english.cuda()
                batch_english_pos = batch_english_pos.cuda()
                batch_french = batch_french.cuda()
                batch_french_pos = batch_french_pos.cuda()

            batches.append((batch_english, batch_english_pos, batch_french, batch_french_pos))
        random.shuffle(batches)
        return batches

    def word_positions(self, line):
        result = []
        pos = 1
        for word in line:
            result.append(pos)
            if pos > self.max_pos: self.max_pos = pos
            if not (len(word) > 2 and word[-2:] == '@@'): pos += 1
        return result
        
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

    def bpe_to_sentence(self, sequence):
        sentence = []
        separated_word = False
        for token in sequence:
            if separated_word and not "@@" in sequence:
                sentence.append(token)
            elif separated_word:
                sentence[-1] = sentence[-1] + token.split("@@")[0]
                if not "@@" in token:
                    separated_word = False
            else:
                sentence.append(token.split("@@")[0])
                separated_word = True

        return sentence
