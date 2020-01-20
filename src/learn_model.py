# dataloader
from src.utils import load_datasets
# pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
# method related
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
# evaluation related
from src.utils import accuracy
from sklearn.metrics import f1_score
# general
import logging as log
import numpy as np


def set_selection_method(config):
    """
    Given the configuration settings, this function instantiates the configured
    feature selection method initialized with the preset parameters.

    @param config: the configuration file object loaded using yaml.safe_load()
    @return: an object that implements the TransformerMixin class (with fit(),
    fit_transform() and transform() methods).
    """
    transformer = None

    selection_cfg = config.get("feature_selection", None)
    if selection_cfg:
        method_name = selection_cfg.get("method", None)

        # checks for RandomizedLasso
        if method_name == "LinearSVC":
            p = selection_cfg.get("parameters", None)
            if p:
                transformer = LinearSVC()
            else:
                transformer = LinearSVC()

    return transformer


def set_scorer_functions(scorers):
    scores = []
    for score in scorers:
        if score == 'accuracy':
            scores.append((score, accuracy))

    return scores


def set_optimization_params(opt):
    params = {}
    for key, item in opt.items():
        # checks if the item is a list with numbers (ignores cv and n_jobs
        # params)
        if isinstance(item, list) and (len(item) == 3) and type(item) is int:
            # create linear space for each parameter to be tuned
            params[key] = np.linspace(item[0], item[1], num=item[2],
                                      endpoint=True)

        elif isinstance(item, list) and type(item) is str:
            print(key, item)
            params[key] = item

    return params


def optimize_model(estimator, X_train, y_train, params, scores, folds, verbose,
                   n_jobs):
    clf = None
    for score_name, score_func in scores:
        log.info("Tuning hyper-parameters for %s" % score_name)

        log.debug(params)
        log.debug(scores)

        clf = GridSearchCV(estimator, params, loss_func=score_func,
                           cv=folds, verbose=verbose, n_jobs=n_jobs)

        clf.fit(X_train, y_train)

        log.info("Best parameters set found on development set:")
        log.info(clf.best_params_)

    return clf.best_estimator_


def set_learning_method(config, X_train, y_train):
    """
    Instantiates the sklearn's class corresponding to the value set in the
    configuration file for running the learning method.

    @param config: configuration object
    @return: an estimator with fit() and predict() methods
    """
    estimator = None

    learning_cfg = config.get("learning", None)
    if learning_cfg:
        p = learning_cfg.get("parameters", None)
        o = learning_cfg.get("optimize", None)
        scorers = \
            set_scorer_functions(learning_cfg.get("scorer", ['mae', 'rmse']))

        method_name = learning_cfg.get("method", None)
        if method_name == "SVR":
            if o:
                tune_params = set_optimization_params(o)
                estimator = optimize_model(SVR(), X_train, y_train,
                                           tune_params,
                                           scorers,
                                           o.get("cv", 5),
                                           o.get("verbose", True),
                                           o.get("n_jobs", 1))

            elif p:
                estimator = SVR(C=p.get("C", 10),
                                epsilon=p.get('epsilon', 0.01),
                                kernel=p.get('kernel', 'rbf'),
                                degree=p.get('degree', 3),
                                gamma=p.get('gamma', 0.0034),
                                tol=p.get('tol', 1e-3),
                                verbose=False)
            else:
                estimator = SVR()

        elif method_name == "SVC":
            if o:
                tune_params = set_optimization_params(o)
                estimator = optimize_model(SVC(), X_train, y_train,
                                           tune_params,
                                           scorers,
                                           o.get('cv', 5),
                                           o.get('verbose', True),
                                           o.get('n_jobs', 1))

            elif p:
                estimator = SVC(C=p.get('C', 1.0),
                                kernel=p.get('kernel', 'rbf'),
                                degree=p.get('degree', 3),
                                gamma=p.get('gamma', 0.0),
                                coef0=p.get('coef0', 0.0),
                                tol=p.get('tol', 1e-3),
                                verbose=p.get('verbose', False))
            else:
                estimator = SVC()

    return estimator, scorers


def fit_predict(config, X_train, y_train, X_test=None, y_test=None,
                ref_thd=None):
    '''
    Uses the configuration dictionary settings to train a model using the
    specified training algorithm. If set, also evaluates the trained model
    in a test set. Additionally, performs feature selection and model parameters
    optimization.

    @param config: the configuration dictionary obtained parsing the
    configuration file.
    @param X_train: the np.array object for the matrix containing the feature
    values for each instance in the training set.
    @param y_train: the np.array object for the response values of each instance
    in the training set.
    @param X_test: the np.array object for the matrix containing the feature
    values for each instance in the test set. Default is None.
    @param y_test: the np.array object for the response values of each instance
    in the test set. Default is None.
    '''
    # sets the selection method
    transformer = set_selection_method(config)
    selector = SelectFromModel(transformer)

    # sets learning algorithm and runs it over the training data
    estimator, scorers = set_learning_method(config, X_train, y_train)

    clf = Pipeline([
        ('feature_selection', selector),
        ('classification', estimator)
    ])
    log.info("Running learning algorithm %s" % str(estimator))
    clf.fit(X_train, y_train)
    log.info("Selected features: %s",
             str(np.extract(selector.get_support(), np.arange(1, 18))))

    if (X_test is not None) and (y_test is not None):
        log.info("Predicting unseen data using the trained model...")
        y_hat = estimator.predict(X_test[:, selector.get_support()])
        log.info("Evaluating prediction on the test set...")
        for scorer_name, scorer_func in scorers:
            v = scorer_func(y_test, y_hat)
            log.info("%s = %s" % (scorer_name, v))

        log.info("Customized scores: ")
        try:
            log.info("F1 score: = %s" % f1_score(y_test, y_hat))
        except:
            pass
        try:
            log.info("Accuracy: = %s" % accuracy(y_test, y_hat))
        except:
            pass

        predict_file = config.get("predict", None)
        if predict_file:
            with open(predict_file, 'w') as f:
                for hyp, ref in zip(y_test, y_hat):
                    f.write("%s\t%s\n" % (hyp, ref))

    return clf


def learn_model(config):
    '''
    Checks for mandatory parameters, opens input files and performs the
    learning steps.
    '''
    # check if the mandatory parameters are set in the config file
    x_train_path = config.get("x_train", None)
    if not x_train_path:
        msg = "'x_train' option not found in the configuration file. \
        The training dataset is mandatory."
        raise Exception(msg)

    y_train_path = config.get("y_train", None)
    if not y_train_path:
        msg = "'y_train' option not found in the configuration file. \
        The training dataset is mandatory."
        raise Exception(msg)

    learning = config.get("learning", None)
    if not learning:
        msg = "'learning' option not found. At least one \
        learning method must be set."
        raise Exception(msg)

    log.info("Opening input files ...")
    log.debug("X_train: %s" % x_train_path)
    log.debug("y_train: %s" % y_train_path)
    x_test_path = config.get("x_test", None)
    y_test_path = config.get("y_test", None)
    if x_test_path and y_test_path:
        log.debug("X_test: %s" % x_test_path)
        log.debug("y_test: %s" % y_test_path)

    # open feature and response files
    X_train, y_train, X_test, y_test = \
        load_datasets(x_train_path, y_train_path, x_test_path, y_test_path)

    # fits training data and predicts the test set using the trained model
    return fit_predict(config, X_train, y_train, X_test, y_test,
                       config.get("ref_thd", None))
