# sentiment_classifier.py

import argparse
import sys
import time
from models import *
from sentiment_data import *
from evaluator import *
from typing import List
import random, torch, numpy as np

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    parser.add_argument('--glove_path', type=str, default=None, help='path to the glove.6B.300d.txt file (optional)')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings (for FFNN)')
    parser.add_argument('--n_hidden_units', type=int, default=300, help='dimension of hidden units (for FFNN)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Set up overall seed
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load train, dev, and test exs and index the words.
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)
    test_exs_words_only = read_blind_sst_examples(args.blind_test_path)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs_words_only)) + " train/dev/test examples")

    # Train and evaluate
    start_time = time.time()
    model = train_feedforward_neural_net(args, train_exs, dev_exs)
    print("\n=====Train Accuracy=====")
    evaluate(model, train_exs)
    print("=====Dev Accuracy=====")
    evaluate(model, dev_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))

    # Write the test set output
    if args.run_on_test:
        # load up the vocabulary
        with open("data/vocab.txt", "r") as f:
            vocab = [word.strip() for word in f.readlines()]
        indexing_sentiment_examples(test_exs_words_only, vocabulary=vocab, UNK_idx=1)
        
        all_preds = []
        eval_batch_iterator = SentimentExampleBatchIterator(test_exs_words_only, batch_size=32, PAD_idx=0, shuffle=False) # hard-coded batch size and PAD_idx
        eval_batch_iterator.refresh()
        batch_data = eval_batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, _ = batch_data
            preds = model.batch_predict(batch_inputs, batch_lengths=batch_lengths)
            all_preds += preds
            batch_data = eval_batch_iterator.get_next_batch()
        test_exs_predicted = [SentimentExample(ex.words, all_preds[ex_idx]) for ex_idx, ex in enumerate(test_exs_words_only)]
        write_sentiment_examples(test_exs_predicted, args.test_output_path)
