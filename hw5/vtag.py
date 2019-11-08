import sys

from hmmlib import HMM


def main(args):
    train_path, test_path = args[1], args[2]
    hmm = HMM(with_oov=True, lambd=1.)

    with open(train_path) as fin:
        hmm.train_supervised(fin)
        hmm.normalize_counts()

    with open(test_path) as fin:
        test_tokens, test_tags = hmm.read_hmm_file(fin)

        predicted_test_tags = hmm.decode_viterbi(test_tokens)

        predicted_test_tags_fb, *_ = hmm.decode_forward_backward(test_tokens)

        assert len(test_tokens) == len(predicted_test_tags)
        assert len(test_tokens) == len(predicted_test_tags_fb)

        hmm.evaluate_perplexity(test_tokens, test_tags)
        hmm.evaluate_accuracy(test_tokens, test_tags, predicted_test_tags, 'Viterbi')
        hmm.evaluate_accuracy(test_tokens, test_tags, predicted_test_tags_fb, 'posterior')

    with open('test-output', 'w') as fout:
        hmm.write_hmm_file(fout, test_tokens, predicted_test_tags_fb)


if __name__ == '__main__':
    main(sys.argv)
