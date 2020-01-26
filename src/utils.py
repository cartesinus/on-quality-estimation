import numpy as np


def load_datasets(x_train_path: str, y_train_path: str, x_test_path: str,
                  y_test_path):
    x_train = np.loadtxt(x_train_path)
    y_train_real = np.loadtxt(y_train_path)
    y_train = np.array([1 if el > 0 else 0 for el in y_train_real])

    x_test = np.loadtxt(x_test_path)
    y_test_real = np.loadtxt(y_test_path)
    y_test = np.array([1 if el > 0 else 0 for el in y_test_real])

    return x_train, y_train, x_test, y_test


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
