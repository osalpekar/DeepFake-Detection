import numpy as np

def parse_confmat(data):
    """
    Parses the Confusion Matrix for True Positive, False Positive, True
    Negative, and False Negative
    """
    tp = data[0][0]
    fp = data[0][1]
    fn = data[1][0]
    tn = data[1][1]
    return tp, fp, fn, tn

def compute_precision(confmat):
    """
    Computes the Precision given the Confusion Matrix
    """
    tp, fp, fn, tn = parse_confmat(confmat)
    return float(tp / (tp + fp))

def compute_recall(confmat):
    """
    Computes the Recall given the Confusion Matrix
    """
    tp, fp, fn, tn = parse_confmat(confmat)
    return float(tp / (tp + fn))

def compute_f1(confmat):
    """
    Computes the F1 Score given the Confusion Matrix
    """
    precision = compute_precision(confmat)
    recall = compute_recall(confmat)
    return (2 * precision * recall) / (precision + recall)
