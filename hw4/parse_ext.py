"""
CKY parser based on Jason Eisner's semiring implementation.
"""
import itertools
import sys
import collections
import numpy as np
from typing import Dict, Tuple, List
import argparse


class Semiring:
    """
    Generalized functions for CKY.
    """
    @staticmethod
    def weight(prob):
        raise NotImplementedError

    @staticmethod
    def inv_weight(weight):
        raise NotImplementedError

    @staticmethod
    def zero():
        raise NotImplementedError

    @staticmethod
    def plus(a, b):
        raise NotImplementedError

    @staticmethod
    def times(a, b):
        raise NotImplementedError


class Recognizer(Semiring):
    """
    Semiring for recognizing a parse (if a rule can be parsed).
    Weight = {true, false}
    """
    @staticmethod
    def weight(prob):
        return True

    @staticmethod
    def inv_weight(weight):
        return weight

    @staticmethod
    def zero():
        return False

    @staticmethod
    def plus(a, b):
        return a | b

    @staticmethod
    def times(a, b):
        return a & b


class Viterbi(Semiring):
    """
    Semiring for finding min weight (best) parse.
    Weight = -log2(Prob)
    """
    @staticmethod
    def weight(prob):
        return -np.log2(prob)

    @staticmethod
    def inv_weight(weight):
        return weight

    @staticmethod
    def zero():
        return np.infty

    @staticmethod
    def plus(a, b):
        return min(a, b)

    @staticmethod
    def times(a, b):
        return a + b


class InvViterbi(Viterbi):
    """
    Semiring for finding max weight (worst) parse.
    Weight = log2(Prob)
    """
    @staticmethod
    def weight(prob):
        return np.log2(prob)

    @staticmethod
    def inv_weight(weight):
        return -weight


class Inside(Semiring):
    """
    Semiring for finding total weight.
    Weight = log2(Prob)
    """
    @staticmethod
    def weight(prob):
        return np.log(prob)

    @staticmethod
    def inv_weight(weight):
        return -weight / np.log(2)

    @staticmethod
    def zero():
        return -np.infty

    @staticmethod
    def plus(a, b):
        return np.logaddexp(a, b)

    @staticmethod
    def times(a, b):
        return a + b


class GrammarRule:
    def __init__(self, weight, lhs, rhs):
        self.lhs = lhs  # LHS rule name.
        self.weight = weight  # Rule weight.
        self.rhs = list(rhs)  # RHS rule names.


class Grammar:
    def __init__(self, rules):
        # Traverse rules.
        self.non_terms: List[GrammarRule] = []
        self.inv_pre_terms: Dict[List[GrammarRule]] = collections.defaultdict(list)
        for rule in rules:
            if len(rule.rhs) > 1:
                self.non_terms.append(rule)
            elif len(rule.rhs) == 1:
                self.inv_pre_terms[rule.rhs[0]].append(rule)

    @staticmethod
    def read_from_file(file):
        rules = []
        for line in file:
            line, *_ = line.split('#', 1)  # Strip comments.
            parts = line.strip().split()
            if len(parts) >= 3:  # Ignore rules in incorrect format.
                prob, lhs, *rhs = parts
                rule = GrammarRule(float(prob), lhs, rhs)
                rules.append(rule)

        return Grammar(rules)


class OutputNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        if len(self.children) > 0:
            parts = itertools.chain(['(', self.name, ' '],
                                    [' '.join([repr(child) for child in self.children])],
                                    [')'])
            return ''.join(parts)
        else:
            return self.name


class CKYParser:
    def __init__(self, mode, grammar_path):
        # Set parse mode
        self.semiring = {'RECOGNIZER': Recognizer, 'BEST-PARSE': Viterbi,
                         'WORST-PARSE': InvViterbi, 'TOTAL-WEIGHT': Inside}.get(mode)
        # Read in grammar
        with open(grammar_path) as fin:
            self.grammar = Grammar.read_from_file(fin)

    def parse_sentence(self, sentence):
        """
        Parses a sentence
        :param sentence: string sentence to parse
        :return:
        """
        sentence = sentence.split()
        cky_weight: Dict[Tuple[int, int], Dict[str, np.float64]] = collections.defaultdict(
            lambda: collections.defaultdict(self.semiring.zero))
        cky_back: Dict[Tuple[int, int], Dict[str, Tuple]] = collections.defaultdict(dict)

        for j, token in enumerate(sentence, 1):
            for rule in self.grammar.inv_pre_terms[token]:
                old_weight = cky_weight[j - 1, j][rule.lhs]
                new_weight = self.semiring.weight(rule.weight)
                cky_weight[j - 1, j][rule.lhs] = self.semiring.plus(old_weight, new_weight)
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    for rule in self.grammar.non_terms:
                        assert len(rule.rhs) == 2
                        a, b = rule.rhs
                        c = rule.lhs
                        if a in cky_weight[i, k] and b in cky_weight[k, j]:
                            old_weight = cky_weight[i, j][c]
                            new_weight = self.semiring.times(cky_weight[i, k][a], cky_weight[k, j][b])
                            new_weight = self.semiring.times(self.semiring.weight(rule.weight), new_weight)
                            new_weight = self.semiring.plus(old_weight, new_weight)
                            if old_weight != new_weight:
                                cky_weight[i, j][c] = new_weight
                                cky_back[i, j][c] = (k, a, b)

        parse_weight = cky_weight[0, len(sentence)].get('ROOT')
        back_pointers = cky_back
        return parse_weight, back_pointers

    def get_semiring(self):
        return self.semiring

    def generate_tree(self, sentence, backpointers):
        tokens = sentence.split()
        return self._generate_tree(tokens, backpointers, 0, len(tokens), "ROOT")

    def _generate_tree(self, tokens, backptrs, i, j, l):
        result_node = OutputNode(l)
        if l in backptrs[i, j]:
            # Non-terminal case
            k, a, b = backptrs[i, j][l]
            result_node.add_child(self._generate_tree(tokens, backptrs, i, k, a))
            result_node.add_child(self._generate_tree(tokens, backptrs, k, j, b))
        elif i == j - 1:
            # Pre-terminal case
            result_node.add_child(OutputNode(tokens[i]))
        return result_node


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse a sentence with a given grammar")
    parser.add_argument("mode",
                        choices=["RECOGNIZER", "BEST-PARSE", "WORST-PARSE", "TOTAL-WEIGHT"],
                        help="Mode of the parser")
    parser.add_argument("grammar", help="Path to grammar file")
    parser.add_argument("sentences", help="Path to file containing the sentences to parse")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    cky_parser = CKYParser(args.mode, args.grammar)

    with open(args.sentences) as fin:
        sentences = fin.readlines()

    for sent in sentences:
        # Parse sentence
        parse_weight, parse_backptrs = cky_parser.parse_sentence(sent)

        if args.mode == 'RECOGNIZER':
            print(parse_weight is not None)
        elif args.mode == 'BEST-PARSE' or args.mode == 'WORST-PARSE':
            if parse_weight is not None:
                best_tree = cky_parser.generate_tree(sent, parse_backptrs)
                print('%.3f\t%s' % (cky_parser.get_semiring().inv_weight(parse_weight), best_tree))
            else:
                print('NOPARSE')
        else:  # mode == 'TOTAL-WEIGHT'
            if parse_weight is not None:
                print('%.3f' % cky_parser.get_semiring().inv_weight(parse_weight))
            else:
                print('NOPARSE')
