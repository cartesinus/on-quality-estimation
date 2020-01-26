from typing import List
from statistics import mean
# import tensorflow_hub as hub
import numpy as np
# import tensorflow_text
import pandas as pd
from subprocess import Popen, PIPE
from nltk import ngrams
import os
import logging as log
import re


def no_tokens(text: str) -> float:
    '''
        Return number of tokens in text (split by ' ' ).
        Used in following features:
        f1: number of tokens in the source sentence
        f2: number of tokens in the target sentence
    '''
    text_len = len(text.split(' ')) - 1
    return float(text_len) if text_len > 0 else float(0)


def text_avg_len(text: str) -> float:
    '''
        Return mean length of input text (split by ' ').
        Used in following features:
        f3: average source token length
    '''
    return mean([len(a) for a in text.split(' ')])


def get_lm_prob(sentence: str, srilm_path: str, lm_path: str) -> float:
    '''
        Return logprob for sentence computed with srilm ngram.
        Used in following features:
        f4: LM probability of source sentence (f1011)
        f5: LM probability of target sentence (f1012)
    '''
    with open('sentence.tmp', 'w') as sent_f:
        sent_f.write(sentence)

    log.debug('Computing logprob for input sentence.')
    process = Popen([srilm_path + '/ngram', '-lm', lm_path, '-order', '3',
                     '-debug', '1', '-ppl', 'sentence.tmp'], stdout=PIPE,
                    stderr=PIPE)
    stdout, stderr = process.communicate()
    logprob = stdout
    match = re.search(r'logprob= (.*?) ppl=', str(stdout))

    if match:
        logprob_match = match.group()
        logprob = float(logprob_match.replace('logprob= ', '').replace(' ppl=', ''))
    else:
        logprob = 0.0
    os.remove('sentence.tmp')

    return logprob


# f6: # of target word within the target hypothesis (f1015)
# f7: avg. # of translations per source word in the sentence
# f8: avg. # of trans. per source word in the sentence with inverse frequency
global_quantile = {}


def get_quantile_frequency(sentence: str, ngram_fp: str, ngram_size: int,
                           quantile: int) -> float:
    '''
        Compute percentage of ngrams in the quantile.

        Used in following features:
        f9: % of unigrams in quartile 1 of frequency in source language
        f10: % of unigrams in quartile 4 of frequency in source language
        f11: % of bigrams in quartile 1 of frequency in source language
        f12: % of bigrams in quartile 4 of frequency in source language
        f13: % of trigrams in quartile 1 of frequency in source language
        f14: % of trigrams in quartile 4 of frequency in source language
    '''
    log.debug('Computing quantile %s for ngrams=%s' % (quantile, ngram_size))
    ngram_sum = 0
    ngram_freq = []
    if ngram_size in global_quantile and quantile in global_quantile[ngram_size]:
        ngram_freq = global_quantile[ngram_size][quantile]
    else:
        with open(ngram_fp, 'r') as ngram_file:
            for line in ngram_file:
                ngram, freq = line.split('\t')
                if len(ngram.split(' ')) == ngram_size:
                    ngram_sum += int(freq)
                    ngram_freq.append(freq)
            cutoff = np.quantile(ngram_freq, quantile*0.25, interpolation='lower')
            log.debug('Cutoff = %s' % cutoff)
            global_quantile[ngram_size] = {}
            global_quantile[ngram_size][quantile] = cutoff

    log.debug('Computing ngram frequency in given quantile.')
    in_quantile_cnt = 0
    sentence_ngrams = ngrams(sentence.split(), ngram_size)
    for sentence_ngram in sentence_ngrams:
        ngram_s = " ".join(list(sentence_ngram))
        with open(ngram_fp, 'r') as ngram_file:
            for line in ngram_file:
                counter_ngram, freq = line.split('\t')
                if counter_ngram == ngram_s and freq <= cutoff:
                    log.debug('Found ngram: %s' % counter_ngram)
                    in_quantile_cnt += 1
                    break

    sent_ngram_size = len(list(ngrams(sentence.split(), ngram_size)))
    log.debug("No. of ngrams in quantile = %s" % in_quantile_cnt)
    log.debug("No. of ngrams in input sentence = %s" % sent_ngram_size)
    if in_quantile_cnt > 0 and sent_ngram_size > 0:
        percent_of_ngrams = float(in_quantile_cnt / sent_ngram_size)
    else:
        percent_of_ngrams = 0.0
    log.debug('Percentage of ngrams in quantile = %s' % percent_of_ngrams)
    return percent_of_ngrams


# f15: % of unigrams in the source sentence seen in a corpus


def no_punctuations(text: str) -> float:
    '''
        f16: # of punctuations in source
        f17: # of punctuations in target
    '''
    punct_ = [',', '.', ':']
    return float(len(list(filter(lambda x: x in punct_, text.split(' ')))))


# def get_use_similarity(path: str, src_column: str,
#                       trg_column: str) -> List[float]:
#    '''
#        Compute similarity score column for file stated in input path.
#        This method use Universal Sentence Encoder.
#    '''
#    embed = hub.load("https://tfhub.dev/google/"
#                     "universal-sentence-encoder-multilingual/3")
#
#    df = pd.read_csv(path, delimiter='\t')
#
#    return [np.inner(embed(src), embed(trg))[0][0] for src, trg in
#            zip(df['en-US'], df['de-DE'])]


def extract_features(input_file: str, srilm_path: str,
                     src_lm_path: str, trg_lm_path: str, trg_ncount_path: str) -> List[float]:
    '''
        Compute similarity score column for file stated in input path.
        This method use Universal Sentence Encoder.
    '''
    df = pd.read_csv(input_file, delimiter='\t')

    return [[no_tokens(src),                                      # 1
             no_tokens(trg),                                      # 2
             text_avg_len(src),                                   # 3
             get_lm_prob(src, srilm_path, src_lm_path),           # 4
             get_lm_prob(trg, srilm_path, trg_lm_path),           # 5
             0.0, 0.0, 0.0,                                       # 6-8
             get_quantile_frequency(src, trg_ncount_path, 1, 1),  # 9
             get_quantile_frequency(src, trg_ncount_path, 1, 4),  # 10
             get_quantile_frequency(src, trg_ncount_path, 2, 1),  # 11
             get_quantile_frequency(src, trg_ncount_path, 2, 4),  # 12
             get_quantile_frequency(src, trg_ncount_path, 3, 1),  # 13
             get_quantile_frequency(src, trg_ncount_path, 3, 4),  # 13
             0.0,                                                 # 15
             no_punctuations(src),                                # 16
             no_punctuations(trg)                                 # 17
             ] for src, trg in
            zip(df['src'], df['trg'])]
