import os
import pickle
from collections import Counter
from tqdm import tqdm

def add_unk(original_sentences, threshold, ext_sentences=None):
    #use original sentences and unk low frequency words
    if ext_sentences is None: 
        counts = Counter([item for sublist in original_sentences for item in sublist])
        unk_words = list({k:counts[k] for k in counts if counts[k] <= threshold})
    #use external sentences and unk unknown words
    else:
        counts = Counter([item for sublist in ext_sentences for item in sublist])
        unk_words = list({k:counts[k] for k in counts})
    for line, sentence in enumerate(tqdm(original_sentences)):
        for index, word in enumerate(sentence):
            if word in unk_words and ext_sentences is None:
                original_sentences[line][index] = '-UNK-'
            elif word not in unk_words and ext_sentences is not None:
                original_sentences[line][index] = '-UNK-'
    return original_sentences
    

def read_data(english_file, french_file, unk=False, threshold=1, ttype='training', eng_data=None, fre_data=None):
    english_sentences = []
    french_sentences = []
    with open(english_file, 'r', encoding='utf8') as engf, open(french_file, 'r', encoding='utf8') as fref:
        for line in engf:
            english_sentences.append(["NULL"] + line.split())
        for line in fref:
            french_sentences.append(line.split())
    
    #unk cases
    if unk:
        print("Adding the -UNK- token to the data.")
        englishname = ttype +'_'+ str(threshold)+'_unk.e'
        frenchname = ttype +'_'+ str(threshold)+'_unk.f'
        #load if file is found
        if os.path.isfile(englishname):
            with open (englishname, 'rb') as eng:
                english_sentences = pickle.load(eng)
        else:
            english_sentences = add_unk(english_sentences, threshold, eng_data)
            with open(englishname, 'wb') as eng:
                pickle.dump(english_sentences, eng)
        print("English data complete.")
        
        
        if os.path.isfile(frenchname):
            with open(frenchname, 'rb') as fre:
                french_sentences = pickle.load(fre)
        else:
            french_sentences = add_unk(french_sentences, threshold, fre_data)
            with open(frenchname, 'wb') as fre:
                pickle.dump(french_sentences, fre)             
        print("French data complete")          
        
    assert len(english_sentences) == len(french_sentences), 'data mismatch'
    return list(zip(english_sentences, french_sentences))