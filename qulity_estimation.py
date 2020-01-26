#!/usr/bin/env python3
# encoding: utf-8

# general
import logging as log
import sys
import yaml
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
# model save and load
from joblib import dump, load
# model learning
from src.learn_model import learn_model
# model inference
from src.inference import inference_from_file
# feature extraction
from src.feature_extraction import extract_features


__version__ = '0.1.1'

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

        parser.add_argument("-v", "--verbose", dest="verbose", action="count",
                            help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version',
                            version=__version__)
        parser.add_argument("-c", "--config", action="store", dest="config",
                            help="path to the configuration file (YAML file).")
        parser.add_argument("-b", "--srilm_path", action="store", dest="srilm_path",
                            help="path to where srilm binaries are.")
        parser.add_argument("-l", "--src_lm_path", action="store", dest="src_lm_path",
                            help="path to source lm model (arpa format).")
        parser.add_argument("-m", "--trg_lm_path", action="store", dest="trg_lm_path",
                            help="path to target lm model (arpa format).")
        parser.add_argument("-n", "--trg_ncount_path", action="store", dest="trg_ncount_path",
                            help="path to ngram-count model (arpa format).")
        parser.add_argument("-t", "--train", dest="train", action="store_true",
                            help="learn model defined in config file.")
        parser.add_argument("-f", "--extract_features", dest="feature_extraction",
                            action="store_true", help="extract features from input.")
        parser.add_argument("-p", "--inference", dest="inference", action="store_true",
                            help="load model stated in config file on input.")
        parser.add_argument("-i", "--input", dest="input", action="store",
                            help="input file (tsv with two columnts: src, trg).")
        parser.add_argument("-o", "--output", dest="output", action="store",
                            help="write program output to file.")

        # Process arguments
        args = parser.parse_args()

        if args.verbose:
            log.basicConfig(level=log.DEBUG)
        else:
            log.basicConfig(level=log.INFO)

        # opens the config file (if available)
        cfg_path = args.config
        if cfg_path:
            cfg = None
            with open(cfg_path, "r") as cfg_file:
                cfg = yaml.safe_load(cfg_file.read())

            model_file_path = cfg.get("model", None)

            if args.train:
                model = learn_model(cfg)
                dump(model, model_file_path)
            elif args.inference:
                model = load(model_file_path)
                inference_from_file(model, args.input)

        elif args.feature_extraction:
            if not args.srilm_path:
                srilm_path = os.getenv('SRILM_PATH', None)
            if not srilm_path:
                log.info('You must provide SRILM path, either using'
                         '--srilm_path parameter or by setting $SRILM_PATH'
                         'enviroment variable.')
                return 0

            if not args.src_lm_path:
                log.info('You must provide src LM path with --src_lm_path parameter')
                return 0
            if not args.trg_lm_path:
                log.info('You must provide trg LM path with --trg_lm_path parameter')
                return 0

            features = extract_features(args.input, srilm_path, args.src_lm_path,
                                        args.trg_lm_path, args.trg_ncount_path)
            if args.output:
                with open(args.output, 'w') as outfile:
                    for line in features:
                        outfile.write("\t".join([str(x) for x in line]))
                        outfile.write("\n")
            else:
                for line in features:
                    print("\t".join([str(x) for x in line]))

    except KeyboardInterrupt:
        # handle keyboard interrupt ###
        return 0

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-v")

    sys.exit(main())
