import itertools
import sys
import collections
import numpy as np
from typing import Dict, Tuple, List


class Semiring:
    @staticmethod
    def weight(prob):
        raise NotImplementedError

    @staticmethod
    def log_prob(weight):
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


class Viterbi(Semiring):
    @staticmethod
    def weight(prob):
        return -np.log2(prob)

    @staticmethod
    def log_prob(weight):
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


class MinViterbi(Viterbi):
    @staticmethod
    def weight(prob):
        return np.log2(prob)

    @staticmethod
    def log_prob(weight):
        return -weight


class Inside(Semiring):
    @staticmethod
    def weight(prob):
        return np.log2(prob)

    @staticmethod
    def log_prob(weight):
        return -weight

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

    def parse(self, sentence, semiring):
        cky_weight: Dict[Tuple[int, int], Dict[str, np.float64]] = collections.defaultdict(
            lambda: collections.defaultdict(semiring.zero))
        cky_back: Dict[Tuple[int, int], Dict[str, Tuple]] = collections.defaultdict(dict)

        for j, token in enumerate(sentence, 1):
            for rule in self.inv_pre_terms[token]:
                old_weight = cky_weight[j - 1, j][rule.lhs]
                new_weight = semiring.weight(rule.weight)
                cky_weight[j - 1, j][rule.lhs] = semiring.plus(old_weight, new_weight)
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    for rule in self.non_terms:
                        assert len(rule.rhs) == 2
                        a, b = rule.rhs
                        c = rule.lhs
                        if a in cky_weight[i, k] and b in cky_weight[k, j]:
                            old_weight = cky_weight[i, j][c]
                            new_weight = semiring.times(cky_weight[i, k][a], cky_weight[k, j][b])
                            new_weight = semiring.times(semiring.weight(rule.weight), new_weight)
                            new_weight = semiring.plus(old_weight, new_weight)
                            if old_weight != new_weight:
                                cky_weight[i, j][c] = new_weight
                                cky_back[i, j][c] = (k, a, b)

        retval = cky_weight[0, len(sent)].get('ROOT')
        return retval, cky_back


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


def generate_tree(tokens, backptrs, i, j, l):
    result_node = OutputNode(l)
    if l in backptrs[i, j]:
        # Non-terminal case
        k, a, b = backptrs[i, j][l]
        result_node.add_child(generate_tree(tokens, backptrs, i, k, a))
        result_node.add_child(generate_tree(tokens, backptrs, k, j, b))
    elif i == j - 1:
        # Pre-terminal case
        result_node.add_child(OutputNode(tokens[i]))
    return result_node


if __name__ == '__main__':
    assert len(sys.argv) == 4
    mode, grammar_path, sents_path = sys.argv[1:]

    sring = {'RECOGNIZER': Viterbi, 'BEST-PARSE': Viterbi,
             'WORST-PARSE': MinViterbi, 'TOTAL-WEIGHT': Inside}.get(mode)

    assert sring is not None

    with open(grammar_path) as fin:
        grammar = Grammar.read_from_file(fin)

    with open(sents_path) as fin:
        for row in fin:
            sent = row.strip().split()
            parse_weight, parse_backptrs = grammar.parse(sent, sring)

            if mode == 'RECOGNIZER':
                print(parse_weight is not None)
            elif mode == 'BEST-PARSE' or mode == 'WORST-PARSE':
                if parse_weight is not None:
                    best_tree = generate_tree(sent, parse_backptrs, 0, len(sent), 'ROOT')
                    print('%.3f\t%s' % (sring.log_prob(parse_weight), best_tree))
                else:
                    print('NOPARSE')
            else:  # mode == 'TOTAL-WEIGHT'
                if parse_weight is not None:
                    print('%.3f' % sring.log_prob(parse_weight))
                else:
                    print('NOPARSE')
