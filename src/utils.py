#import graphviz
import numpy as np
from operator import itemgetter


def load_datasets(x_train_path: str, y_train_path: str, x_test_path: str, \
                  y_test_path):
    x_train = np.loadtxt(x_train_path)
    y_train_real = np.loadtxt(y_train_path)
    y_train = np.array([1 if el > 0 else 0 for el in y_train_real])

    x_test = np.loadtxt(x_test_path)
    y_test_real = np.loadtxt(y_test_path)
    y_test = np.array([1 if el > 0 else 0 for el in y_test_real])

    return x_train, y_train, x_test, y_test


def plot_tree(clf):
    f_names = [
        'number of tokens in the source sentence',
        'number of tokens in the target sentence',
        'average source token length',
        'LM probability of source sentence',
        'LM probability of target sentence',
        '# of target word within the target hypothesis',
        'avg. # of translations per source word in the sentence',
        'avg. # of trans. per source word in the sentence with inverse frequency',
        '% of unigrams in quartile 1 of frequency in source language',
        '% of unigrams in quartile 4 of frequency in source language',
        '% of bigrams in quartile 1 of frequency in source language',
        '% of bigrams in quartile 4 of frequency in source language',
        '% of trigrams in quartile 1 of frequency in source language',
        '% of trigrams in quartile 4 of frequency in source language',
        '% of unigrams in the source sentence seen in a corpus',
        '# of punctuations in source',
        '# of punctuations in target'
    ]

    t_names = ['OK', 'BAD']

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=f_names,
                                    class_names=t_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("quality-estimation")


def evaluate(clf, X_dev, y_dev):
    '''Return accuracy of input clf model.'''
    tp = 0

    for pred, gt in zip(clf.predict(X_dev), y_dev):
        if pred == gt:
            tp += 1

    return tp/1000


def accuracy(ref, hyp):
    '''Return accuracy between reference and hypothesys.'''
    tp = 0

    for pred, gt in zip(hyp, ref):
        if pred == gt:
            tp += 1

    return tp/ref.size


#data_path = "/home/cartesinus/projects/quality-estimation/data/sentence-level/"
#X, y = load_data(data_path + "/features/en_de/sentence_features/train.nmt.features",
#                 data_path + "/sentence_level/en_de/train.nmt.hter")

#X_f_selection = [itemgetter(0,1,2,15,16)(x) for x in X]
#y_f_selection = y, because this is 0 / 1 expected result column only

#X_dev, y_dev = load_data(data_path + "/features/en_de/sentence_features/dev.nmt.features",
#                         data_path + "/sentence_level/en_de/dev.nmt.hter")
#X_dev_f_selection = [itemgetter(0,1,2,15,16)(x) for x in X_dev]
#y_dev_f_selection = y_dev, because this is only expected label
