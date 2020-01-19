Overview
========
This repository allow build models for machine translation (MT) quality estimation (QE).

The data consists of:
    - English-German WMT18 sentences on the IT domain translated by in-house encoder-decoder attention-based NMT system (13,442 training and 1,000 development sentences). After running `./scripts/download-data.sh` data will be downloaded to `data/sentence-level/features/en_de`. The usual 17 features used in WMT12-17 is considered for the baseline system. This system uses SVM regression with an RBF kernel, as well as grid search algorithm for the optimisation of relevant parameters.


Running
=======
The program takes only one input parameter, the configuration file. For example:

```
./learn_model.py config/svc.yaml
```


Available algorithms
====================
All of available methods are taken from sklearn, so it is fairly easey to add other
as well, but currently these are "supported":

* SVM
The parameters exposed in the "Parameters" section of the configuration file are:
    - C
    - coef0
    - kernel
    - degree
    - gamma
    - tol
    - verbose

Documentation about these parameters is available at
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC


Feature selection
=================
To set up a feature selection algorithm add the "feature_selection" section to the 
configuration file. This section is independent of the "learning" section:

```
feature_selection:
    method: LinearSVC
    parameters:
        cv: 10

learning:
    ...
```

Currently, the following feature selection algorithms are available:

* Linear Support Vector Classification. The exposed parameters are:

    - penalty (default=’l2’)
    - loss (default=’squared_hinge’)
    - dual (default=True)
    - tol (default=1e-4)
    - C (default=1.0)
    - fit_intercept (default=True)
    - intercept_scaling (default=1)
    - max_iterint (default=1000)

These parameters and the method are documented at:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html


Sentence-level Baseline Features
================================
Following features are used in WMT12-17 quality estimation shared task and are considered as baseline features:

1. number of tokens in the source sentence
2. number of tokens in the target sentence
3. average source token length
4. LM probability of source sentence
5. LM probability of target sentence
6. number of occurrences of the target word within the target hypothesis (averaged for all words in the hypothesis - type/token ratio)
7. average number of translations per source word in the sentence (as given by IBM 1 table thresholded such that prob(t|s) > 0.2)
8. average number of translations per source word in the sentence (as given by IBM 1 table thresholded such that prob(t|s) > 0.01) weighted by the inverse frequency of each word in the source corpus
9. percentage of unigrams in quartile 1 of frequency (lower frequency words) in a corpus of the source language (SMT training corpus)
10. percentage of unigrams in quartile 4 of frequency (higher frequency words) in a corpus of the source language
11. percentage of bigrams in quartile 1 of frequency of source words in a corpus of the source language
12. percentage of bigrams in quartile 4 of frequency of source words in a corpus of the source language
13. percentage of trigrams in quartile 1 of frequency of source words in a corpus of the source language
14. percentage of trigrams in quartile 4 of frequency of source words in a corpus of the source language
15. percentage of unigrams in the source sentence seen in a corpus (SMT training corpus)
16. number of punctuation marks in the source sentence
17. number of punctuation marks in the target sentence
