import random
import bisect
import itertools
import collections


class OutputNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def show_tree(self):
        if len(self.children) > 0:
            parts = itertools.chain(['(', self.name, ' '],
                                    [' '.join([child.show_tree() for child in self.children])],
                                    [')'])
            return ''.join(parts)
        else:
            return self.name

    def show_tokens(self):
        if len(self.children) > 0:
            return ' '.join(child.show_tokens() for child in self.children)
        else:
            return self.name


class StopNode(OutputNode):
    def __init__(self):
        super().__init__('...')


class GrammarRule:
    def __init__(self, weight, lhs, rhs):
        self.lhs = lhs  # LHS rule name.
        self.weight = weight  # Rule weight.
        self.rhs = list(rhs)  # RHS rule names.


class Grammar:
    def __init__(self, rules, seed=None):
        self.random_state = random.Random(seed)

        # Traverse rules.
        self.non_terms = collections.defaultdict(list)  # Storage map for non-terminals.
        for rule in rules:
            self.non_terms[rule.lhs].append(rule)

        assert 'ROOT' in self.non_terms

    @staticmethod
    def read_from_file(file, seed=None):
        rules = []
        for line in file:
            line, *_ = line.split('#', 1)  # Strip comments.
            parts = line.strip().split()
            if len(parts) >= 3:  # Ignore rules in incorrect format.
                weight, lhs, *rhs = parts
                rule = GrammarRule(float(weight), lhs, rhs)
                rules.append(rule)

        return Grammar(rules, seed=seed)

    def generate_tree(self, max_non_terms=None):
        output_root = OutputNode('ROOT')
        non_terms_cnt = 0

        # We will traverse the grammar with breadth-first search (BFS).
        queue = [('ROOT', output_root)]
        while len(queue) > 0:
            # Top element from the queue is always a non-terminal.
            (top, output_parent), *queue = queue
            rules = self.non_terms[top]
            # Choose a random rule according to rules' weights.
            cum_weights = list(itertools.accumulate([rule.weight for rule in rules]))
            cum_sum = cum_weights[-1]
            random_rule_x = self.random_state.uniform(0, cum_sum)
            random_rule_i = min(bisect.bisect(cum_weights, random_rule_x), len(cum_weights) - 1)
            random_rule = rules[random_rule_i]
            for rhs in random_rule.rhs:
                if rhs in self.non_terms:
                    if max_non_terms is None or non_terms_cnt < max_non_terms:
                        # Increase the counter of processed non-terminals.
                        non_terms_cnt += 1
                        output_child = OutputNode(rhs)
                        output_parent.add_child(output_child)
                        queue.append((rhs, output_child))
                    else:
                        stop_child = StopNode()
                        output_parent.add_child(stop_child)
                else:
                    output_parent.add_child(OutputNode(rhs))

        return output_root


if __name__ == '__main__':
    with open('grammar.gr') as grammar_file:
        grammar = Grammar.read_from_file(grammar_file, seed=0)
        output_tree = grammar.generate_tree(max_non_terms=None)
        print('Tree:', output_tree.show_tree())
        print('Tokens:', output_tree.show_tokens())
