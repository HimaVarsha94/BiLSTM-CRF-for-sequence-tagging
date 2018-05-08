from __future__ import division

import argparse
from io import open
import math
import os

from future.builtins import range
from future.utils import iterkeys, iteritems
from sklearn import metrics

def compute_acc(actual, predicted):
    """
    Computes the accuracy of your predictions, using 0.5 as a cutoff.

    Note that these inputs are lists, not dicts; they assume that actual and predicted are in the same order.

    Parameters (here and below):
        actual: a list of the actual labels
        predicted: a list of your predicted labels
    """
    num = len(actual)
    acc = 0.
    for i in range(num):
        if round(actual[i], 0) == round(predicted[i], 0):
            acc += 1.
    acc /= num
    return acc


def compute_avg_log_loss(actual, predicted):
    """
    Computes the average log loss of your predictions.
    """
    num = len(actual)
    loss = 0.

    for i in range(num):
        p = predicted[i] if actual[i] > .5 else 1. - predicted[i]
        loss -= math.log(p)
    loss /= num
    return loss


def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc


def compute_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1
def compute_auc(all_target, all_pred):
    # fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)

def evaluate_metrics(actual, predicted):
    """
    This computes and returns a dictionary of notable evaluation metrics for your predicted labels.
    """
    acc = compute_acc(actual, predicted)
    avg_log_loss = compute_avg_log_loss(actual, predicted)
    auroc = compute_auroc(actual, predicted)
    F1 = compute_f1(actual, predicted)
    auc = compute_auc(actual, predicted)

    return {'accuracy': acc, 'avglogloss': avg_log_loss, 'auroc': auroc, 'F1': F1 ,'sklearn_auc':auc}

if __name__ == '__main__':
    main()
