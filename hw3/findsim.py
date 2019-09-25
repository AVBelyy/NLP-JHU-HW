#!/usr/bin/env python3
"""
findsim.py
JHU NLP HW3
Name: Anton Belyi
Email: abel@jhu.edu
Term: Fall 2019
"""
import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional  # Type annotations.

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

log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

TOPK = 10  # Not really a variable, but a static constant.


def read_into_data_structure(vector_file: Path) -> Any:  # 'Any' matches any type.
    """Parse the file into memory in some usable, efficient format."""
    log.info("Reading data into usable format...")

    inv_vocab = {}
    vocab = []
    vecs = []

    with vector_file.open() as fin:
        n_words, dim = map(int, fin.readline().strip().split())

        for i, line in enumerate(fin):
            word, *vec = line.strip().split()
            vec = np.array([float(v) for v in vec])
            vec /= np.linalg.norm(vec, ord=2)  # Divide by norm to simplify further computation of cosine similarity.
            vocab.append(word)
            inv_vocab[word] = i
            vecs.append(vec)

        vecs = np.vstack(vecs)

        assert vecs.shape == (n_words, dim)

    log.info("- done.")

    return vecs, vocab, inv_vocab


def do_work(
        data_structure: Any, word1: str, word2: Optional[str], word3: Optional[str]
) -> List[str]:
    """Perform your actual computation here."""
    log.info("Doing real work...")

    vecs, vocab, inv_vocab = data_structure

    assert word1 in inv_vocab

    ix1 = inv_vocab[word1]
    if word2 is not None and word3 is not None:
        assert word2 in inv_vocab
        assert word3 in inv_vocab

        ix2, ix3 = inv_vocab[word2], inv_vocab[word3]
    else:
        ix2, ix3 = ix1, ix1

    search_vec = vecs[ix1] - vecs[ix2] + vecs[ix3]
    search_vec /= np.linalg.norm(search_vec, ord=2)
    search_vec = search_vec.reshape(-1, 1)

    cosine_simils = vecs.dot(-search_vec).T[0]

    top_indices = np.argpartition(cosine_simils, TOPK + 3)[:TOPK + 3]
    top_indices = [ix for ix in top_indices if ix not in (ix1, ix2, ix3)]
    # Sort top indices w.r.t. their cosine values
    top_indices = sorted(top_indices, key=cosine_simils.__getitem__)
    top_indices = top_indices[:TOPK]
    top_words = [vocab[ix] for ix in top_indices]

    log.info("- done.")
    return top_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if len([x for x in (args.minus, args.plus) if x is not None]) == 1:
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args


def main():
    args = parse_args()
    data = read_into_data_structure(args.embeddings)
    words = do_work(data, args.word, args.minus, args.plus)
    print(" ".join(words))


if __name__ == "__main__":
    main()
