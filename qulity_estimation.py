#!/usr/bin/env python3
# encoding: utf-8

# general
import logging as log
import sys
import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter
# model save and load
from joblib import dump, load
# model learning
from src.learn_model import learn_model
# model inference
from src.inference import inference_from_file
# feature extraction
# from src.feature_extraction import extract_features


__all__ = []
__version__ = '0.1.0'

DEFAULT_SEP = "\t"
DEBUG = 1


def main(argv=None):  # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    try:
        # Setup argument parser
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)

        parser.add_argument("-c", "--config", action="store", dest="config",
                            help="path to the configuration file (YAML file).")
        parser.add_argument("-v", "--verbose", dest="verbose", action="count",
                            help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version',
                            version=__version__)
        parser.add_argument("-t", "--train", dest="train", action="store_true",
                            help="learn model defined in config file.")
        parser.add_argument("-i", "--input", dest="input", action="store",
                            help="load input file to run inference on it.")
        parser.add_argument("-p", "--inference", dest="inference",
                            action="store_true",
                            help="load model stated in config file on input.")
        parser.add_argument("-f", "--extract_features",
                            dest="feature_extraction", action="store_true",
                            help="load model stated in config file on input.")

        # Process arguments
        args = parser.parse_args()

        cfg_path = args.config

        if args.verbose:
            log.basicConfig(level=log.DEBUG)
        else:
            log.basicConfig(level=log.INFO)

        # opens the config file
        cfg = None
        with open(cfg_path, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file.read())

        model_file_path = cfg.get("model", None)
        if args.train:
            model = learn_model(cfg)
            if model_file_path:
                dump(model, model_file_path)
        elif args.inference:
            if model_file_path:
                model = load(model_file_path)
                inference_from_file(model, args.input)
#        elif args.feature_extraction:
#            extract_features(args.input)

    except KeyboardInterrupt:
        # handle keyboard interrupt ###
        return 0

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-v")

    sys.exit(main())
