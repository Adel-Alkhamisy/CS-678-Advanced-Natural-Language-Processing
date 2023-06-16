from typing import List
from sentiment_data import *

def evaluate(classifier, exs: List[SentimentExample], return_metrics: bool=False):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :param return_metrics: set to True if returning the stats
    :return: None (but prints output)
    """
    all_labels = []
    all_preds = []

    eval_batch_iterator = SentimentExampleBatchIterator(exs, batch_size=32, PAD_idx=0, shuffle=False) # hard-coded batch size and PAD_idx
    eval_batch_iterator.refresh()
    batch_data = eval_batch_iterator.get_next_batch()
    while batch_data is not None:
        batch_inputs, batch_lengths, batch_labels = batch_data
        all_labels += list(batch_labels)

        preds = classifier.batch_predict(batch_inputs, batch_lengths=batch_lengths)
        all_preds += list(preds)
        batch_data = eval_batch_iterator.get_next_batch()

    if return_metrics:
        acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)
        return acc, prec, rec, f1
    else:
        calculate_metrics(all_labels, all_preds, print_only=True)


def calculate_metrics(golds: List[int], predictions: List[int], print_only: bool=False):
    """
    Calculate evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Returns accuracy, precision, recall, and F1.

    :param golds: gold labels
    :param predictions: pred labels
    :param print_only: set to True if printing the stats without returns
    :return: accuracy, precision, recall, and F1 (all floating numbers), or None (when print_only is True)
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    acc = float(num_correct) / num_total
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0

    print("Accuracy: %i / %i = %f" % (num_correct, num_total, acc))
    print("Precision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
          + "; Recall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
          + "; F1 (harmonic mean of precision and recall): %f" % f1)

    if not print_only:
        return acc, prec, rec, f1
