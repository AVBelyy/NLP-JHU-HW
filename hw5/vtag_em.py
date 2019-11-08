import sys

from hmmlib import HMM


def main(args):
    train_path, test_path, raw_path = args[1], args[2], args[3]
    num_iterations = 10
    hmm = HMM(with_oov=True, lambd=1.)

    with open(train_path) as fin:
        hmm.train_supervised(fin)
        train_dict = set(hmm.uniq_tokens)

    with open(raw_path) as fin:
        raw_tokens, raw_tags = hmm.read_hmm_file(fin, has_tags=False)

        hmm.initialize_em(raw_tokens)
        raw_dict = hmm.uniq_tokens - train_dict

    with open(test_path) as fin:
        test_tokens, test_tags = hmm.read_hmm_file(fin)

        predicted_test_tags = hmm.decode_viterbi(test_tokens)
        hmm.evaluate_perplexity(test_tokens, test_tags)
        hmm.evaluate_accuracy_em(raw_dict, test_tokens, test_tags, predicted_test_tags, 'Viterbi')

        for t in range(num_iterations + 1):
            raw_perplexity = hmm.do_em_step(raw_tokens)
            print('Iteration %d: Model perplexity per untagged raw word: %.3f' % (t, raw_perplexity))

            predicted_test_tags = hmm.decode_viterbi(test_tokens)
            hmm.evaluate_perplexity(test_tokens, test_tags)
            hmm.evaluate_accuracy_em(raw_dict, test_tokens, test_tags, predicted_test_tags, 'Viterbi')

        predicted_test_tags_fb, *_ = hmm.decode_forward_backward(test_tokens)

    with open('test-output', 'w') as fout:
        hmm.write_hmm_file(fout, test_tokens, predicted_test_tags_fb)


if __name__ == '__main__':
    main(sys.argv)
