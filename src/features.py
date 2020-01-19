from typing import List
from statistics import mean
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import pandas as pd


def no_tokens(text: str) -> float:
    '''
        Return number of tokens in text (split by ' ' ).
        Used in following features:
        f1: number of tokens in the source sentence
        f2: number of tokens in the target sentence
    '''
    text_len = len(text.split('')) - 1
    return float(text_len) if text_len > 0 else float(0)


def text_avg_len(text: str) -> float:
    '''
        Return mean length of input text (split by ' ').
        Used in following features:
        f3: average source token length
    '''
    return mean([len(a) for a in text.split(' ')])

# f4: LM probability of source sentence
# f5: LM probability of target sentence
# f6: # of target word within the target hypothesis
# f7: avg. # of translations per source word in the sentence
# f8: avg. # of trans. per source word in the sentence with inverse frequency
# f9: % of unigrams in quartile 1 of frequency in source language
# f10: % of unigrams in quartile 4 of frequency in source language
# f11: % of bigrams in quartile 1 of frequency in source language
# f12: % of bigrams in quartile 4 of frequency in source language
# f13: % of trigrams in quartile 1 of frequency in source language
# f14: % of trigrams in quartile 4 of frequency in source language
# f15: % of unigrams in the source sentence seen in a corpus


def no_punctuations(text: str) -> float:
    '''
        f16: # of punctuations in source
        f17: # of punctuations in target
    '''
    punct_ = [',', '.', ':']
    return float(len(list(filter(lambda x: x in punct_, text.split(' ')))))


def get_use_similarity(path: str, src_column: str,
                       trg_column: str) -> List[float]:
    '''
        Compute similarity score column for file stated in input path.
        This method use Universal Sentence Encoder.
    '''
    embed = hub.load("https://tfhub.dev/google/"
                     "universal-sentence-encoder-multilingual/3")

    df = pd.read_csv(path, delimiter='\t')

    return [np.inner(embed(src), embed(trg))[0][0] for src, trg in
            zip(df['en-US'], df['de-DE'])]


def do_everything(path: str, src_column: str,
                       trg_column: str) -> List[float]:
    '''
        Compute similarity score column for file stated in input path.
        This method use Universal Sentence Encoder.
    '''
    df = pd.read_csv(path, delimiter='\t')

    return [[no_tokens(src), no_tokens(trg), text_avg_len(src), no_punctuations(src), no_punctuations(trg)] for src, trg in
            zip(df['en-US'], df['de-DE'])]
