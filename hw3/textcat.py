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
    parser.add_argument("train1_file", type=Path, help="location of the first training corpus")
    parser.add_argument("train2_file", type=Path, help="location of the second training corpus")
    parser.add_argument("train1_prior", type=float, help="prior probability of the first training corpus", nargs="?")
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
    elif args.mode == "TEST" and not args.train1_prior:
        parser.error("No prior probability for train1 specified.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model1_path = get_model_filename(args.smoother, args.lexicon, args.train1_file)
    model2_path = get_model_filename(args.smoother, args.lexicon, args.train2_file)

    if args.mode == TRAIN:
        log.info("Training...")
        lm = LanguageModel.make(args.smoother, args.lexicon)
        lm.set_vocab_size(args.train1_file, args.train2_file)
        lm.train(args.train1_file)
        lm.save(destination=model1_path)
        lm.train(args.train2_file)
        lm.save(destination=model2_path)
    elif args.mode == TEST:
        log.info("Testing...")
        log_probs_models = []

        for model_path in (model1_path, model2_path):
            log.info(f"Estimating file log-likelihoods under {model_path.name}.")
            log_probs_models.append([])
            lm = LanguageModel.load(model_path)
            for test_file in args.test_files:
                log_prob = lm.file_log_prob(test_file)
                log_probs_models[-1].append(log_prob)

        pred_labels_cnts = np.array([0, 0])
        for test_file, (log_p_x_model1, log_p_x_model2) in zip(args.test_files, zip(*log_probs_models)):
            log_p_x_model1 += np.log(args.train1_prior)
            log_p_x_model2 += np.log(1 - args.train1_prior)

            if log_p_x_model1 >= log_p_x_model2:
                pred_label = args.train1_file.name
                pred_labels_cnts[0] += 1
            else:
                pred_label = args.train2_file.name
                pred_labels_cnts[1] += 1

            print(f"{pred_label}\t{test_file.name}")

        pred_labels_ratios = 100 * pred_labels_cnts / pred_labels_cnts.sum()
        print(f"{pred_labels_cnts[0]} files were more probably {args.train1_file.name} ({pred_labels_ratios[0]:.2f}%)")
        print(f"{pred_labels_cnts[1]} files were more probably {args.train2_file.name} ({pred_labels_ratios[1]:.2f}%)")

    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

