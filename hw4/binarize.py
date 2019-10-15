import sys
import copy
import argparse
import collections
import numpy as np

from pathlib import Path
from typing import List, Dict, Set, Tuple


GrammarRule = collections.namedtuple('GrammarRule', ['weight', 'lhs', 'rhs'])


class Grammar:
    def __init__(self, rules):
        self.rules: List[GrammarRule] = rules
        self.non_terms_set: Set[str] = {rule.lhs for rule in rules}

    @staticmethod
    def read_from_file(file):
        rules = []
        for line in file:
            line, *_ = line.split('#', 1)  # Strip comments.
            parts = line.strip().split()
            if len(parts) >= 3:  # Ignore rules in incorrect format.
                prob, lhs, *rhs = parts
                rule = GrammarRule(np.log(float(prob)), lhs, tuple(rhs))
                rules.append(rule)

        return Grammar(rules)


class CNFGrammar:
    def __init__(self, old_grammar, root_symbol):
        self.counter = 0
        self.old_grammar = grammar

        self.n_walks = 0
        self.n_steps = 0

        self.non_terms: Dict[str, Dict[Tuple, float]] = \
            collections.defaultdict(lambda: collections.defaultdict(lambda: -np.inf))

        # Construct unit productions graph.
        self.up_graph: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        self.up_chains: Dict[str, Dict[str, Dict[Tuple, float]]] = \
            collections.defaultdict(lambda: collections.defaultdict(dict))
        self.construct_unit_prod_graph()

        # Transform non-unary rules.
        for rule in old_grammar.rules:
            if len(rule.rhs) > 2:
                self.transform_long_rule(rule)
            elif len(rule.rhs) == 2:
                self.transform_binary_rule(rule)
            elif rule.rhs[0] not in self.old_grammar.non_terms_set:
                self.non_terms[rule.lhs][rule.rhs] = np.logaddexp(self.non_terms[rule.lhs][rule.rhs], rule.weight)

        # Transform unary rules.
        self.frozen_non_terms = copy.deepcopy(self.non_terms)
        for rule in old_grammar.rules:
            if len(rule.rhs) == 1 and rule.rhs[0] in self.old_grammar.non_terms_set:
                self.transform_unary_rule(rule)

        # Check correctness of weights.
        for lhs, rules in self.non_terms.items():
            total_log_prob = np.logaddexp.reduce(list(rules.values()))
            assert abs(total_log_prob) < 1e-6, "Grammar weights are incorrect for non-terminal %s!" % lhs

        # Delete unused rules.
        unused_non_terms = set(self.non_terms.keys()) - {root_symbol}
        queue: List[str] = [root_symbol]

        while len(queue) > 0:
            cur, queue = queue[0], queue[1:]

            for rule in self.non_terms[cur].keys():
                for rhs in rule:
                    if rhs in unused_non_terms:
                        unused_non_terms.remove(rhs)
                        queue.append(rhs)

        for lhs in unused_non_terms:
            del self.non_terms[lhs]

    def output_to_file(self, outfile):
        # Output rules.
        for lhs, rules in self.non_terms.items():
            for rhs, weight in rules.items():
                outfile.write('%.16f\t%s\t%s\n' % (np.exp(weight), lhs, ' '.join(rhs)))

    def make_new_nonterm(self):
        self.counter += 1
        return 'X%d' % self.counter

    def construct_unit_prod_graph(self):
        up_leaves = set()

        def walk(path, cur, weight):
            if weight < -20:
                return
            root, rest = path[0], path[1:]

            self.up_chains[cur][root][rest] = weight
            self.n_steps += 1
            for up_next, up_weight in self.up_graph[cur]:
                    walk(path + (cur,), up_next, weight + up_weight)

        for rule in self.old_grammar.rules:
            if len(rule.rhs) == 1 and rule.rhs[0] in self.old_grammar.non_terms_set:
                self.up_graph[rule.rhs[0]].append((rule.lhs, rule.weight))
                up_leaves.add(rule.rhs[0])

        for leaf in up_leaves:
            for next_leaf, next_weight in self.up_graph[leaf]:
                self.n_walks += 1
                walk((leaf,), next_leaf, next_weight)

    def abstract_terminals(self, rhs):
        # The input is RHS of a rule.
        # The output is RHS with abstracted away terminals.
        # That is, for each terminal `term` in RHS, a separate rule X_i -> term is created and X_i is used instead.

        new_rhs = []
        for token in rhs:
            if token in self.old_grammar.non_terms_set:
                new_rhs.append(token)
            else:
                new_var = self.make_new_nonterm()
                self.non_terms[new_var][token, ] = 0.0
                new_rhs.append(new_var)
        return tuple(new_rhs)

    def transform_unary_rule(self, rule):
        # The input is:
        # A -> B
        lhs, rhs = rule.lhs, rule.rhs[0]
        lhs_weight_prior = self.up_chains[lhs][rhs][()]

        # Process possible direct extensions of B
        for rule_rhs, rule_weight in self.frozen_non_terms[rhs].items():
            self.non_terms[lhs][rule_rhs] = np.logaddexp(self.non_terms[lhs][rule_rhs], rule_weight + lhs_weight_prior)

        # Process possible indirect (using other unit prod chains) extensions of B
        for cur, paths in self.up_chains[rhs].items():
            for _, rhs_weight_prior in paths.items():
                for rule_rhs, rule_weight in self.frozen_non_terms[cur].items():
                    delta_weight = rule_weight + lhs_weight_prior + rhs_weight_prior
                    self.non_terms[lhs][rule_rhs] = np.logaddexp(self.non_terms[lhs][rule_rhs], delta_weight)

    def transform_binary_rule(self, rule):
        # The input is:
        # A -> B C (w.p. p)

        new_rhs = self.abstract_terminals(rule.rhs)

        self.non_terms[rule.lhs][new_rhs] = np.logaddexp(self.non_terms[rule.lhs][new_rhs], rule.weight)

    def transform_long_rule(self, rule):
        # The input is:
        # A -> B C D E F (w.p. p)

        # Let's create new vars X1, X2 and X3.
        new_vars = [self.make_new_nonterm() for _ in range(len(rule.rhs) - 2)]

        new_rhs = self.abstract_terminals(rule.rhs)

        # This will produce:
        # A -> B X1 (w.p. p)
        self.non_terms[rule.lhs][new_rhs[0], new_vars[0]] = rule.weight

        # These will produce:
        # X1 -> C X2 (w.p. 1.0)
        # X2 -> D X3 (w.p. 1.0)
        for lhs, a, b in zip(new_vars[:-1], new_rhs[1:-2], new_vars[1:]):
            self.non_terms[lhs][a, b] = 0.0

        # This will produce:
        # X3 -> E F (w.p. 1.0)
        self.non_terms[new_vars[-1]][new_rhs[-2:]] = 0.0


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('grammar_file', type=Path, help='PCFG grammar file name')
    parser.add_argument('--outfile', type=Path, help='output file name')
    parser.add_argument('--root', type=str, help='grammar root symbol name', default='ROOT')
    parser_args = parser.parse_args()
    assert parser_args.grammar_file.is_file()
    return parser_args


if __name__ == '__main__':
    args = parse_args()

    with args.grammar_file.open() as fin:
        grammar = Grammar.read_from_file(fin)
        cnf_grammar = CNFGrammar(grammar, args.root)

    fout = None
    try:
        if args.outfile:
            fout = args.outfile.open('w')
        else:
            fout = sys.stdout

        cnf_grammar.output_to_file(fout)
    finally:
        if args.outfile and fout is not None:
            fout.close()
