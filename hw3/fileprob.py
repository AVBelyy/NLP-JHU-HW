#!/usr/bin/env python3
"""
Computes the log probability of the sequence of tokens in file,
according to a trigram model.  The training source is specified by
the currently open corpus, and the smoothing method used by
prob() is polymorphic.
"""
import argparse
import logging
from pathlib import Path

try:
    # Numpy is your friend. Not *using* it will make your program so slow.
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import numpy as np
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


from Probs import LanguageModel

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, train_file: Path) -> Path:
    return Path(f"{smoother}_{lexicon.name}_{train_file.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "smoother",
        type=str,
        help="""Possible values: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the "1" in add1/backoff_add1 can be replaced with any real λ >= 0
   the "1" in loglinear1 can be replaced with any C >= 0 )
""",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word vector file; only used in the loglinear model",
    )
    parser.add_argument("train_file", type=Path, help="location of the training corpus")
    parser.add_argument("test_files", type=Path, nargs="*")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()

    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and not args.test_files:
        parser.error("No test files specified.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path = get_model_filename(args.smoother, args.lexicon, args.train_file)

    if args.mode == TRAIN:
        log.info("Training...")
        lm = LanguageModel.make(args.smoother, args.lexicon)

        lm.train(args.train_file)
        lm.save(destination=model_path)
    elif args.mode == TEST:
        log.info("Testing...")
        lm = LanguageModel.load(model_path)
        # We use natural log for our internal computations and that's
        # the kind of log-probability that fileLogProb returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        log.info("Printing file log-likelihoods.")
        total_log_prob = 0.0
        for test_file in args.test_files:
            log_prob = lm.file_log_prob(test_file) / np.log(2)
            print(f"{log_prob:g}\t{test_file}")
            total_log_prob += log_prob

        total_tokens = sum(lm.num_tokens(test_file) for test_file in args.test_files)
        print(f"Overall cross-entropy:\t{-total_log_prob / total_tokens:.5f}")
    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

