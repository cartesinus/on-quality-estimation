Overview
========
This repository allow build models for machine translation (MT) quality estimation (QE).
It is clearly Quest++ rip off that I made in order to experiment with 'before BERT' QE.

The data :
    - English-German WMT18 sentences on the IT domain translated by in-house
      encoder-decoder attention-based NMT system (13,442 training and 1,000 development
      sentences)
    - After running `./scripts/download-data.sh` data will be downloaded to
      `data/sentence-level/features/en_de`.
    - The usual 17 features used in WMT12-17 is considered for the baseline system
    - WMT18 QE baseline model was SVM regression with an RBF kernel, with grid search
      algorithm for the optimisation of relevant parameters. I tried to reproduce this
      in config/svc.cfg


Running
=======
The program takes as an input; method, config file and additional parameters.

For example, to train model:
```
./quality_estimation.py --train --config config/svc.yaml
```

To inference model on given input:
```
./quality_estimation.py --inference --config config/svc.yaml --input test.tsv
```

To extract features from tsv file (needed columnt: src and trg):
```
./qulity_estimation.py -v --src_lm_path data/lm.tok.en --trg_lm_path data/lm.tok.de --trg_ncount_path data/ngram-count.de -f -i test.tsv
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
