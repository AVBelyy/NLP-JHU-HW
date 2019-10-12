#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separatiing business logic from other stuff.  (9/19/2019)

import logging
import math
import re
import sys

import numpy as np

from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple, Union
from scipy.misc import logsumexp

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

Zerogram = Tuple[()]
Unigram = Tuple[str]
Bigram = Tuple[str, str]
Trigram = Tuple[str, str, str]
Ngram = Union[Zerogram, Unigram, Bigram, Trigram]
Vector = List[float]

BOS = "BOS"  # special word type for context at Beginning Of Sequence
EOS = "EOS"  # special word type for observed token at End Of Sequence
OOV = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL = "OOL"  # special word type for all Out-Of-Lexicon words
OOV_THRESHOLD = (
    3
)  # minimum number of occurrence for a word to be considered in-vocabulary


def get_tokens(file: Path):
    """Iterate over the tokens, saving a few layers of nesting."""
    with open(file) as corpus:
        for line in corpus:
            for z in line.split():
                yield z
    yield EOS  # Every file implicitly ends with EOS.


class LanguageModel:
    def __init__(self):
        self.tokens: Counter[Ngram] = Counter()  # the c(...) function.
        self.vocab: Optional[Set[str]] = None
        self.progress = 0  # To print progress.

    @classmethod
    def make(cls, smoother: str, lexicon: Path) -> "LanguageModel":
        """Factory pattern: Build the language model you need."""
        r = re.compile("^(.*?)-?([0-9.]*)$")
        m = r.match(smoother)

        lambda_: Optional[float]  # Type annotation only.

        if m is None or not m.lastindex:
            raise ValueError(f"Smoother regular expression failed for {smoother}")
        else:
            smoother_name = m.group(1).lower()
            if m.lastindex >= 2 and len(m.group(2)):
                lambda_arg = m.group(2)
                lambda_ = float(lambda_arg)
            else:
                lambda_ = None

        if lambda_ is None and smoother_name.find("add") != -1:
            raise ValueError(
                f"You must include a non-negative lambda value in smoother name {smoother}"
            )

        if smoother_name == "uniform":
            return UniformLanguageModel()
        elif smoother_name == "add":
            assert lambda_ is not None
            return AddLambdaLanguageModel(lambda_)
        elif smoother_name == "backoff_add":
            assert lambda_ is not None
            return BackoffAddLambdaLanguageModel(lambda_)
        elif smoother_name == "loglin":
            assert lambda_ is not None
            return LogLinearLanguageModel(lambda_, lexicon)
        elif smoother_name == "improved":
            lambda_ = lambda_ or 1.0
            gamma = 0.5
            return LogLinearLanguageModel(lambda_, lexicon, is_improved=True, gamma=gamma)
        else:
            raise ValueError(f"Don't recognize smoother name {smoother_name}")

    def file_log_prob(self, corpus: Path) -> float:
        """Compute the log probability of the sequence of tokens in file.
        NOTE: we use natural log for our internal computation.  You will want to
        divide this number by log(2) when reporting log probabilities.
        """
        log_prob = 0.0
        u, v, x, y = BOS, BOS, BOS, BOS
        for z in get_tokens(corpus):
            if hasattr(self, "log_prob"):
                log_prob += self.log_prob(x, y, z, u, v)
            else:
                prob = self.prob(x, y, z, u, v)
                log_prob += np.log(prob)
            u, v, x, y = v, x, y, z  # Shift over by one position.
        return log_prob

    def set_vocab_size(self, *files: Path) -> None:
        if self.vocab is not None:
            log.warning("Warning: vocabulary already set!")

        word_counts: Counter[str] = Counter()  # count of each word

        for file in files:
            for token in get_tokens(file):
                word_counts[token] += 1
                self.show_progress()
        sys.stderr.write("\n")  # done printing progress dots "...."

        vocab: Set[str] = set(w for w in word_counts if word_counts[w] >= OOV_THRESHOLD)
        vocab |= {  # Union equals
            OOV,
            EOS,
        }  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)

        self.vocab = vocab
        log.info(f"Vocabulary size is {self.vocab_size} types including OOV and EOS")

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def count(self, x: str, y: str, z: str) -> None:
        """Count the n-grams.  In the perl version, this was an inner function.
        For now, I am just using a data member to store the found tri-
        and bigrams.
        """
        self._count_ngram((x, y, z))
        self._count_ngram((y, z))
        self._count_ngram((z,))
        self._count_ngram(())

    def _count_ngram(self, ngram: Ngram) -> None:
        """Count the n-gram; that is, increment its count in the model."""
        self.tokens[ngram] += 1

    def num_tokens(self, corpus: Path) -> int:
        """Give the number of tokens in the corpus, including EOS."""
        return sum(1 for token in get_tokens(corpus))

    def prob(self, x: str, y: str, z: str, u: Optional[str], v: Optional[str]) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("Reimplement this in subclasses!")
        raise NotImplementedError(
            f"{class_name} is not implemented yet. (That's your job!)"
        )

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle

        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            return pickle.load(f)
        log.info(f"Loaded model from {source}")

    def save(self, destination: Path) -> None:
        import pickle

        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def replace_missing(self, token: str) -> str:
        assert self.vocab is not None
        if token not in self.vocab:
            return OOV
        return token

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        log.info(f"Training from corpus {corpus}")
        # The real work:
        # accumulate the type and token counts into the global hash tables.

        # If vocab size has not been set, build the vocabulary from training corpus
        if self.vocab is None:
            self.set_vocab_size(corpus)

        # Clear out any previous training.
        self.tokens = Counter()

        # We save the corpus in memory to a list tokens_list.  Notice that we
        # prepended two BOS at the front of the list and appended an EOS at the end.  You
        # will need to add more BOS tokens if you want to use a longer context than
        # trigram.
        x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
        # count the BOS context

        self.tokens[(x, y)] = 1
        self.tokens[(y,)] = 1  # The syntax for a 1-element tuple in Python

        tokens_list = [x, y]  # the corpus saved as a list
        for z in get_tokens(corpus):
            z = self.replace_missing(z)
            self.count(x, y, z)
            self.show_progress()
            x, y = y, z  # Shift over by 1 word.
            tokens_list.append(z)
        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.tokens[()]} tokens")
        return tokens_list

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


class UniformLanguageModel(LanguageModel):
    def prob(self, x: str, y: str, z: str, u: Optional[str], v: Optional[str]) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str, u: Optional[str], v: Optional[str]) -> float:
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)
        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.
        return (self.tokens[x, y, z] + self.lambda_) / (
                self.tokens[x, y] + self.lambda_ * self.vocab_size
        )


class BackoffAddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str, u: Optional[str], v: Optional[str]) -> float:
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)

        lamV = self.lambda_ * self.vocab_size

        p_hat_z = (self.tokens[z,] + self.lambda_) / (self.tokens[()] + lamV)
        p_hat_zy = (self.tokens[y, z] + lamV * p_hat_z) / (self.tokens[y,] + lamV)
        p_hat_zxy = (self.tokens[x, y, z] + lamV * p_hat_zy) / (self.tokens[x, y] + lamV)

        return p_hat_zxy


class LogLinearLanguageModel(LanguageModel):
    def __init__(self, c: float, lexicon_path: Path, is_improved: Optional[bool] = False, gamma: Optional[float] = None) -> None:
        super().__init__()

        if c < 0:
            log.error(f"C value was {c}")
            raise ValueError("You must include a non-negative c value in smoother name")
        self.c: float = c
        self.vectors: np.ndarray
        self.lexicon: Dict[str, int]
        self.vocab_vectors: np.ndarray
        self.vocab_lexicon: Dict[str, int]
        self.vectors, self.lexicon = self._read_vectors(lexicon_path)
        self.dim: int = self.vectors.shape[0]

        # Base weights
        self.X: np.ndarray = None
        self.Y: np.ndarray = None

        # Added features' weights
        self.logC: np.float32

        self.is_improved: Optional[bool] = is_improved
        self.gamma: Optional[float] = gamma

    def _read_vectors(self, lexicon_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
        """Read word vectors from an external file.  The vectors are saved as
        arrays in a dictionary self.vectors.
        """
        with lexicon_path.open() as f:
            header = f.readline()
            n_words, dim = map(int, header.split())
            vectors: List[np.ndarray] = []
            lexicon: Dict[str, int] = {}
            for i, line in enumerate(f):
                word, *arr = line.split()
                vec = np.array([float(x) for x in arr])
                vectors.append(vec)
                lexicon[i] = word
                lexicon[word] = i

            vec_arr = np.vstack(vectors).T
            assert vec_arr.shape == (dim, n_words)

        return vec_arr, lexicon

    def replace_missing(self, token: str) -> str:
        # substitute out-of-lexicon words with OOL symbol
        assert self.vocab is not None
        if token not in self.vocab:
            token = OOV
        if token not in self.lexicon:
            token = OOL
        return token

    def _log_denom(self, x_idx: int, y_idx: int, u_idx: int, v_idx: int) -> np.ndarray:
        x_vec = self.vocab_vectors[:, [x_idx]]
        y_vec = self.vocab_vectors[:, [y_idx]]

        xXZ = x_vec.T.dot(self.X).dot(self.vocab_vectors)
        yYZ = y_vec.T.dot(self.Y).dot(self.vocab_vectors)

        # Added features
        if self.is_improved:
            u_vec = self.vocab_vectors[:, [u_idx]]
            v_vec = self.vocab_vectors[:, [v_idx]]
            uXZ = (self.gamma ** 2) * u_vec.T.dot(self.X).dot(self.vocab_vectors)
            vXZ = self.gamma * v_vec.T.dot(self.X).dot(self.vocab_vectors)
            cZ = np.array([[self.tokens[t, ] for t in self.vocab_lexicon_list]])
            logcZ = self.logC * np.log(cZ + 1)

            logXYZ = xXZ + yYZ + logcZ + uXZ + vXZ
        else:
            logXYZ = xXZ + yYZ

        return logXYZ

    def set_vocab_size(self, *files: Path):
        super().set_vocab_size(*files)

        # get the subset of the lexicon vectors for the vocabulary
        self.vocab_vectors = np.zeros((self.dim, len(self.vocab)), dtype=self.vectors.dtype)
        self.vocab_lexicon = {}
        self.vocab_lexicon_list = []
        for i, token in enumerate(self.vocab):
            token = self.replace_missing(token)
            token_idx = self.lexicon[token]
            vec = self.vectors[:, token_idx]
            self.vocab_vectors[:, i] = vec
            self.vocab_lexicon[token] = i
            self.vocab_lexicon_list.append(token)

        # To save memory and disk space
        self.vectors = None

    def log_prob(self, x: str, y: str, z: str, u: str, v: str) -> float:
        assert self.vocab is not None
        u = self.replace_missing(u)
        v = self.replace_missing(v)
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)

        u_idx = self.vocab_lexicon[u]
        v_idx = self.vocab_lexicon[v]
        x_idx = self.vocab_lexicon[x]
        y_idx = self.vocab_lexicon[y]
        z_idx = self.vocab_lexicon[z]

        logXYZ = self._log_denom(x_idx, y_idx, u_idx, v_idx)

        return logXYZ[0, z_idx] - logsumexp(logXYZ)

    def prob(self, x: str, y: str, z: str, u: Optional[str], v: Optional[str]) -> float:
        return np.exp(self.log_prob(x, y, z, u, v))

    def objective(self, tokens_list: List[str]):
        result = 0.

        # Calculate sum(log{...}, i=1..N)
        for i in range(4, len(tokens_list)):
            u, v, x, y, z = tokens_list[i - 4], tokens_list[i - 3], tokens_list[i - 2], tokens_list[i - 1], \
                            tokens_list[i]

            result += self.log_prob(x, y, z, u, v)

        # Subtract the regularizer (the frobenius norm)
        result -= self.c * (np.sum(self.X ** 2) + np.sum(self.Y ** 2))

        # Subtract the regularizer for added features
        if self.is_improved:
            result -= self.c * (self.logC ** 2)

        return result / self.N

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        tokens_list = super().train(corpus)
        tokens_list = [BOS, BOS] + tokens_list
        # Train the log-linear model using SGD.

        # Initialize parameters
        self.X = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.Y = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.logC = 0.

        # Optimization hyperparameters
        gamma0 = 0.01  # initial learning rate, used to compute actual learning rate
        epochs = 10  # number of passes

        self.N = len(tokens_list) - 4  # number of training instances
        # ******** COMMENT *********
        # In log-linear model, you will have to do some additional computation at
        # this point.  You can enumerate over all training trigrams as following.
        #
        # for i in range(2, len(tokens_list)):
        #   x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
        #
        # Note1: self.c is the regularizer constant C
        # Note2: You can use self.show_progress() to log progress.
        #
        # **************************

        log.info("Start optimizing.")

        t = 0
        for e in range(epochs):
            for i in range(4, len(tokens_list)):
                u, v, x, y, z = tokens_list[i - 4], tokens_list[i - 3], tokens_list[i - 2], tokens_list[i - 1], \
                                tokens_list[i]

                u = self.replace_missing(u)
                v = self.replace_missing(v)
                x = self.replace_missing(x)
                y = self.replace_missing(y)
                z = self.replace_missing(z)

                u_idx = self.vocab_lexicon[u]
                v_idx = self.vocab_lexicon[v]
                x_idx = self.vocab_lexicon[x]
                y_idx = self.vocab_lexicon[y]
                z_idx = self.vocab_lexicon[z]

                u_vec = self.vocab_vectors[:, [u_idx]]
                v_vec = self.vocab_vectors[:, [v_idx]]
                x_vec = self.vocab_vectors[:, [x_idx]]
                y_vec = self.vocab_vectors[:, [y_idx]]
                z_vec = self.vocab_vectors[:, [z_idx]]

                logXYZ = self._log_denom(x_idx, y_idx, u_idx, v_idx)
                log_p_z_xy = logXYZ - logsumexp(logXYZ)
                p_z_xy = np.exp(log_p_z_xy)

                z_exp_vec = self.vocab_vectors.dot(p_z_xy.T)

                dFdX = np.outer(x_vec, z_vec) - np.outer(x_vec, z_exp_vec) - (2 * self.c / self.N) * self.X
                dFdY = np.outer(y_vec, z_vec) - np.outer(y_vec, z_exp_vec) - (2 * self.c / self.N) * self.Y

                # Added features' gradient updates
                if self.is_improved:
                    cZ = np.array([self.tokens[t, ] for t in self.vocab_lexicon_list])
                    logcZ = np.log(cZ + 1)
                    dFdlogC = logcZ[z_idx] - np.dot(p_z_xy[0], logcZ) - (2 * self.c / self.N) * self.logC
                    dFdX += np.outer(u_vec, z_vec) - np.outer(u_vec, z_exp_vec) - (self.gamma ** 2) * (2 * self.c / self.N) * self.X
                    dFdX += np.outer(v_vec, z_vec) - np.outer(v_vec, z_exp_vec) - self.gamma * (2 * self.c / self.N) * self.X

                gamma = gamma0 / (1 + gamma0 * 2 * self.c * t / self.N)
                self.X += gamma * dFdX
                self.Y += gamma * dFdY

                if self.is_improved:
                    self.logC += gamma * dFdlogC

                t += 1

            log.info(f"epoch {e + 1}: F={self.objective(tokens_list)}")

        log.info(f"Finished training on {self.tokens[()]} tokens")
        return tokens_list  # Not really needed, except to obey typing.
