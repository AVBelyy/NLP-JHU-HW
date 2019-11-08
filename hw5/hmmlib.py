from copy import deepcopy

import numpy as np

from collections import defaultdict


class HMM:
    def __init__(self, with_oov=False, lambd=0.):
        # Set by train_supervised and train_with_em.
        self.transition_counts = None
        self.emission_counts = None
        self.uniq_tokens = None
        self.uniq_tags = None
        self.tag_dict = None
        self.oov_tag_set = None

        # Set by initialize_em.
        self.transition_counts_orig = None
        self.emission_counts_orig = None

        # Set by normalize_counts.
        self.emission_denom_counts = None
        self.transition_denom_counts = None
        self.transition_log_probs = None
        self.emission_log_probs = None
        self.oov_transition_log_probs = None
        self.oov_emission_log_probs = None

        # Set by __init__.
        self.with_oov = int(with_oov)
        self.lambd = lambd

    def initialize_em(self, raw_tokens):
        # Update vocabulary.
        self.uniq_tokens.update(raw_tokens)

        # Copy train_supervised counts.
        self.transition_counts_orig = deepcopy(self.transition_counts)
        self.emission_counts_orig = deepcopy(self.emission_counts)

        # Normalize current counts for the 0th iteration.
        self.normalize_counts()

    def do_em_step(self, raw_tokens):
        # E-step:
        # Obtain new count estimates from raw data.
        _, transition_counts_new, emission_counts_new, raw_perplexity = self.decode_forward_backward(raw_tokens)

        # Merge supervised and new EM counts.
        self.transition_counts = {k: v + self.transition_counts_orig.get(k, 0) for k, v in transition_counts_new.items()}
        self.emission_counts = {k: v + self.emission_counts_orig.get(k, 0) for k, v in emission_counts_new.items()}

        # M-step:
        # Update HMM probabilities based on current estimates.
        self.normalize_counts()

        return raw_perplexity

    def read_hmm_file(self, hmm_file, has_tags=True):
        tokens, tags = [], []

        for line in hmm_file:
            line = line.strip()
            if has_tags:
                token, tag = line.split('/')
                token = token if token in self.uniq_tokens else 'OOV'
                tags.append(tag)
            else:
                token = line
            tokens.append(token)

        return tokens, tags

    @staticmethod
    def write_hmm_file(fout, tokens, tags):
        for test_out in zip(tokens, tags):
            fout.write('%s/%s\n' % test_out)

    def train_supervised(self, train_file):
        self.transition_counts = defaultdict(int)
        self.emission_counts = defaultdict(int)
        self.tag_dict = defaultdict(set)

        prev_tag = None

        uniq_tokens = set()
        uniq_tags = set()

        # Count numerators for MLE estimators
        for i, line in enumerate(train_file):
            token, tag = line.strip().split('/')
            if i > 0:
                self.transition_counts[tag, prev_tag] += 1
            self.emission_counts[token, tag] += 1
            self.tag_dict[token].add(tag)
            uniq_tokens.add(token)
            uniq_tags.add(tag)

            prev_tag = tag

        self.uniq_tokens = uniq_tokens
        self.uniq_tags = uniq_tags

        if self.with_oov:
            self.uniq_tokens.add('OOV')

        self.oov_tag_set = uniq_tags - {'###'}

    def normalize_counts(self):
        self.transition_denom_counts = defaultdict(int)
        self.emission_denom_counts = defaultdict(int)

        for (_, prev_tag), count in self.transition_counts.items():
            self.transition_denom_counts[prev_tag] += count

        for (_, tag), count in self.emission_counts.items():
            self.emission_denom_counts[tag] += count

        self.transition_log_probs = {}
        self.emission_log_probs = {}

        self.oov_transition_log_probs = {}
        self.oov_emission_log_probs = {}

        # In-vocabulary transition probabilities.
        for (tag, prev_tag), num in self.transition_counts.items():
            num = num + self.lambd
            denom = self.transition_denom_counts[prev_tag] + self.lambd * len(self.uniq_tags)
            self.transition_log_probs[tag, prev_tag] = np.log(num) - np.log(denom)

        # In-vocabulary emission probabilities.
        for (token, tag), num in self.emission_counts.items():
            num = num + self.lambd
            denom = self.emission_denom_counts[tag] + self.lambd * len(self.uniq_tokens)
            self.emission_log_probs[token, tag] = np.log(num) - np.log(denom)

        # Out-of-vocabulary transition and emission probabilities.
        log_lambd = np.log(self.lambd) if self.lambd > 1e-9 else -np.infty
        delta_transition_denom = self.lambd * len(self.uniq_tags)
        delta_emission_denom = self.lambd * len(self.uniq_tokens)
        for tag in self.uniq_tags:
            transition_denom = self.transition_denom_counts[tag] + delta_transition_denom
            emission_denom = self.emission_denom_counts[tag] + delta_emission_denom
            self.oov_transition_log_probs[tag] = log_lambd - np.log(transition_denom)
            self.oov_emission_log_probs[tag] = log_lambd - np.log(emission_denom)

        # Sanity check
        # total_emission_log_probs = defaultdict(lambda: -np.infty)
        # for tag in self.uniq_tags:
        #     for token in self.uniq_tokens:
        #         log_prob = self.emission_log_prob(token, tag)
        #         total_emission_log_probs[tag] = np.logaddexp(total_emission_log_probs[tag], log_prob)
        #
        # total_transition_log_probs = defaultdict(lambda: -np.infty)
        # for tag in self.uniq_tags:
        #     for prev_tag in self.uniq_tags:
        #         log_prob = self.transition_log_prob(tag, prev_tag)
        #         total_transition_log_probs[prev_tag] = np.logaddexp(total_transition_log_probs[prev_tag], log_prob)
        #
        # for tag, log_prob in total_emission_log_probs.items():
        #     assert np.abs(log_prob) < 1e-9
        #
        # for tag, log_prob in total_transition_log_probs.items():
        #     assert np.abs(log_prob) < 1e-9

    def emission_log_prob(self, token, tag):
        if (token, tag) in self.emission_log_probs:
            return self.emission_log_probs[token, tag]
        else:
            return self.oov_emission_log_probs[tag]

    def transition_log_prob(self, tag, prev_tag):
        if (tag, prev_tag) in self.transition_log_probs:
            return self.transition_log_probs[tag, prev_tag]
        else:
            return self.oov_transition_log_probs[prev_tag]

    def tag_set(self, token):
        if token in self.tag_dict:
            return self.tag_dict[token]
        else:
            return self.oov_tag_set

    def decode_viterbi(self, test_tokens):
        prev_token = '###'
        prev_best_log_probs = {'###': 0}
        backpointers = []

        for token in test_tokens[1:]:
            cur_best_log_probs = defaultdict(lambda: -np.infty)
            cur_backpointers = {}

            # This is just Dijkstra's algorithm over weighted transition graph, with weight = log(arc-prob).
            for tag in self.tag_set(token):
                for prev_tag in self.tag_set(prev_token):
                    arc_log_prob = self.transition_log_prob(tag, prev_tag) + self.emission_log_prob(token, tag)
                    cur_log_prob = prev_best_log_probs[prev_tag] + arc_log_prob
                    if cur_best_log_probs[tag] < cur_log_prob:
                        cur_best_log_probs[tag] = cur_log_prob
                        cur_backpointers[tag] = prev_tag

            prev_best_log_probs = cur_best_log_probs
            backpointers.append(cur_backpointers)
            prev_token = token

        best_tags = ['###']
        cur_tag = '###'

        for i in range(len(backpointers) - 1, -1, -1):
            cur_tag = backpointers[i][cur_tag]
            best_tags.append(cur_tag)

        return best_tags[::-1]

    def decode_forward_backward(self, test_tokens):
        prev_token = '###'
        all_fwd_log_probs = [{'###': 0}]

        # Do a forward pass.
        for token in test_tokens[1:]:
            prev_fwd_log_probs = all_fwd_log_probs[-1]
            cur_fwd_log_probs = defaultdict(lambda: -np.infty)

            for tag in self.tag_set(token):
                for prev_tag in self.tag_set(prev_token):
                    arc_log_prob = self.transition_log_prob(tag, prev_tag) + self.emission_log_prob(token, tag)
                    delta_fwd_log_prob = prev_fwd_log_probs[prev_tag] + arc_log_prob
                    cur_fwd_log_probs[tag] = np.logaddexp(cur_fwd_log_probs[tag], delta_fwd_log_prob)

            all_fwd_log_probs.append(cur_fwd_log_probs)
            prev_token = token

        log_total_prob = all_fwd_log_probs[-1]['###']
        untagged_perplexity = np.exp(-log_total_prob / (len(test_tokens) - 1))

        best_tags = []
        prev_bck_log_probs = {'###': 0}

        # probability estimates for E step of EM.
        transition_log_cnt_em = defaultdict(lambda: -np.infty)
        emission_log_cnt_em = defaultdict(lambda: -np.infty)

        # Do a backward pass.
        for i in range(len(test_tokens) - 1, 0, -1):
            token = test_tokens[i]
            prev_token = test_tokens[i - 1]
            cur_bck_log_probs = defaultdict(lambda: -np.infty)

            cur_argmax_prob, cur_max_log_prob = None, -np.infty
            for fwd_tag, fwd_log_prob in all_fwd_log_probs[i].items():
                if fwd_tag in prev_bck_log_probs:
                    fwd_bck_log_prob = fwd_log_prob + prev_bck_log_probs[fwd_tag] - log_total_prob
                    if cur_max_log_prob < fwd_bck_log_prob:
                        cur_max_log_prob = fwd_bck_log_prob
                        cur_argmax_prob = fwd_tag
                    emission_log_cnt_em[token, fwd_tag] = np.logaddexp(emission_log_cnt_em[token, fwd_tag],
                                                                       fwd_bck_log_prob)

            best_tags.append(cur_argmax_prob)

            for tag in self.tag_set(token):
                for prev_tag in self.tag_set(prev_token):
                    arc_log_prob = self.transition_log_prob(tag, prev_tag) + self.emission_log_prob(token, tag)
                    delta_bck_log_prob = prev_bck_log_probs[tag] + arc_log_prob
                    cur_bck_log_probs[prev_tag] = np.logaddexp(cur_bck_log_probs[prev_tag], delta_bck_log_prob)
                    transition_log_prob = all_fwd_log_probs[i - 1][prev_tag] + delta_bck_log_prob - log_total_prob
                    transition_log_cnt_em[tag, prev_tag] = np.logaddexp(transition_log_cnt_em[tag, prev_tag],
                                                                        transition_log_prob)

            prev_bck_log_probs = cur_bck_log_probs

        transition_cnt_em = {k: np.exp(v) for k, v in transition_log_cnt_em.items()}
        emission_cnt_em = {k: np.exp(v) for k, v in emission_log_cnt_em.items()}

        return ['###'] + best_tags[::-1], transition_cnt_em, emission_cnt_em, untagged_perplexity

    def evaluate_perplexity(self, true_tokens, true_tags):
        total_log_prob = 0.
        total_count = 0
        for token, tag, prev_tag in zip(true_tokens[1:], true_tags[1:], true_tags[:-1]):
            total_count += 1
            total_log_prob += self.emission_log_prob(token, tag) + self.transition_log_prob(tag, prev_tag)

        # print(total_log_prob, total_count)
        perplexity_per_word = np.exp(-total_log_prob / total_count)
        print('Model perplexity per tagged test word: %.3f' % perplexity_per_word)

    @staticmethod
    def evaluate_accuracy(true_tokens, true_tags, predicted_tags, decoding_type):
        overall_result = [a == b for a, b in zip(true_tags, predicted_tags) if a != '###']
        known_result = [a == b for a, b, c in zip(true_tags, predicted_tags, true_tokens) if a != '###' and c != 'OOV']
        novel_result = [a == b for a, b, c in zip(true_tags, predicted_tags, true_tokens) if a != '###' and c == 'OOV']
        overall_acc = 100. * float(np.mean(overall_result)) if overall_result else 0.
        known_acc = 100. * float(np.mean(known_result)) if known_result else 0.
        novel_acc = 100. * float(np.mean(novel_result)) if novel_result else 0.
        out_results = (decoding_type, overall_acc, known_acc, novel_acc)
        out_str = 'Tagging accuracy (%s decoding): %.2f%% (known: %.2f%% novel: %.2f%%)' % out_results
        print(out_str)

    @staticmethod
    def evaluate_accuracy_em(raw_dict, true_tokens, true_tags, predicted_tags, decoding_type):
        overall_result = [a == b for a, b in zip(true_tags, predicted_tags) if a != '###']
        known_result = [a == b for a, b, c in zip(true_tags, predicted_tags, true_tokens) if a != '###' and c != 'OOV']
        seen_result = [a == b for a, b, c in zip(true_tags, predicted_tags, true_tokens) if a != '###' and c in raw_dict]
        novel_result = [a == b for a, b, c in zip(true_tags, predicted_tags, true_tokens) if a != '###' and c == 'OOV']
        overall_acc = 100. * float(np.mean(overall_result)) if overall_result else 0.
        known_acc = 100. * float(np.mean(known_result)) if known_result else 0.
        seen_acc = 100. * float(np.mean(seen_result)) if seen_result else 0.
        novel_acc = 100. * float(np.mean(novel_result)) if novel_result else 0.
        out_results = (decoding_type, overall_acc, known_acc, seen_acc, novel_acc)
        out_str = 'Tagging accuracy (%s decoding): %.2f%% (known: %.2f%% seen: %.2f%% novel: %.2f%%)' % out_results
        print(out_str)
