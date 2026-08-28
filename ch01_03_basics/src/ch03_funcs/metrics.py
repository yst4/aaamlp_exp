import numpy as np
from collections import Counter

def indicator(y, target_class):
    """
    A function that binarizes labels (1/0) based on the specified `target_class`.
    Args:
        y: list of true values
        target_class:
    Results:
        list of 0/1 sequence.
    """
    return [1 if p == target_class else 0 for p in y]

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        accuracy score
    """
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1

    return correct_counter / len(y_true)


def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results: number of true positives
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        number of true negatives
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    """
    Function to calculate False Positives
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        number of false positives
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    """
    Function to calculate False Negatives
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        number of false negatives
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

"""
# approach 1 (functional style)
def _count_matches(y_true, y_pred, cond):
    return sum(1 for yt, yp in zip(y_true, y_pred) if cond(yt, yp))

def true_positive(y_true, y_pred):
    return _count_matches(y_true, y_pred, lambda yt, yp: yt == 1 and yp == 1)

def true_negative(y_true, y_pred):
    return _count_matches(y_true, y_pred, lambda yt, yp: yt == 0 and yp == 0)

def false_positive(y_true, y_pred):
    return _count_matches(y_true, y_pred, lambda yt, yp: yt == 0 and yp == 1)

def false_negative(y_true, y_pred):
    return _count_matches(y_true, y_pred, lambda yt, yp: yt == 1 and yp == 0)

"""

"""
# apporach 2 (more effective)
def confusion_matrix_counts(y_true, y_pred):
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def true_positive(y_true, y_pred):
    return confusion_matrix_counts(y_true, y_pred)["tp"]

def true_negative(y_true, y_pred):
    return confusion_matrix_counts(y_true, y_pred)["tn"]

def false_positive(y_true, y_pred):
    return confusion_matrix_counts(y_true, y_pred)["fp"]

def false_negative(y_true, y_pred):
    return confusion_matrix_counts(y_true, y_pred)["fn"]

"""

def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy version 2
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        accuracy score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    """
    Function to calculate precision
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        precision score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """
    Function to calculate recall
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred):
    """
    Function to calculate f1 score
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        f1 score
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def tpr(y_true, y_pred):
    """
    Function to calculate tpr
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        tpr/recall
    """
    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    Function to calculate fpr
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp) if (tn + fp) > 0 else 0.0

def log_loss(y_true, y_proba):
    """
    Function to calculate log loss
    Args:
        y_true: list of true values
        y_proba: list of probabilities for 1
    Results:
        overall log loss
    """
    epsilon = 1e-15
    loss = []

    for yt, yp in  zip(y_true, y_proba):
        yp = np.clip(yp, epsilon, 1 - epsilon)
        temp_loss = -1.0 * (
            yt * np.log(yp) + (1 - yt) * np.log(1 - yp)
        )
        loss.append(temp_loss)

    return np.mean(loss)

def macro_precision(y_true, y_pred):
    """
    Function to calculate macro averaged precision
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        macro precision score
    """
    classes = np.unique(y_true)
    precisions = []

    for class_ in classes:
        temp_true = indicator(y_true, class_)
        temp_pred = indicator(y_pred, class_)
        precisions.append(precision(temp_true, temp_pred))

    return np.mean(precisions) if len(precisions) > 0 else 0.0

def micro_precision(y_true, y_pred):
    """
    Function to calculate micro averaged precision
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        micro precision score
    """
    classes = np.unique(y_true)
    total_tp = 0
    total_fp = 0

    for class_ in classes:
        temp_true = indicator(y_true, class_)
        temp_pred = indicator(y_pred, class_)

        total_tp += true_positive(temp_true, temp_pred)
        total_fp += false_positive(temp_true, temp_pred)

    return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

def weighted_precision(y_true, y_pred):
    """
    Function to calculate weighted averaged precision
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        weighted precision score
    """
    classes = np.unique(y_true)
    class_counts = Counter(y_true)
    weighted_p_sum = 0.0

    for class_ in classes:
        temp_true = indicator(y_true, class_)
        temp_pred = indicator(y_pred, class_)

        p = precision(temp_true, temp_pred)
        weighted_p_sum += class_counts[class_] * p

    return weighted_p_sum / len(y_true) if len(y_true) > 0 else 0.0

def weighted_f1(y_true, y_pred):
    """
    Function to calculate weighted f1 score
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    Results:
        weighted f1 score
    """
    classes = np.unique(y_true)
    class_counts = Counter(y_true)
    weighted_f1_sum = 0.0

    for class_ in classes:
        temp_true = indicator(y_true, class_)
        temp_pred = indicator(y_pred, class_)

        f1_score = f1(temp_true, temp_pred)
        weighted_f1_sum += class_counts[class_] * f1_score

    return weighted_f1_sum / len(y_true) if len(y_true) > 0 else 0.0

def pk(y_true,y_pred,k):
    """
    This function calculates precision at k
    for a single sample
    Args:
        y_true: list of values, actual classes
        y_pred: list of values, predicated classes
        k: the value of k
    Results:
        precision at a given value k
    """
    if k == 0:
        return 0

    y_pred =y_pred[:k]
    pred_set = set(y_pred)
    true_set = set(y_true)

    common_values = pred_set.intersection(true_set)
    return len(common_values)/ len(y_pred[:k])

def apk(y_true, y_pred, k):
    """
    This function calculates average precision at k
    for a single sample
    Args:
        y_true: list of values, actual classes
        y_pred: list of values, predicated classes
        k: the value of k
    Results:
        average precision at a given value k
    """
    pk_values =[]
    for i in range(1, k + 1):
        pk_values.append(pk(y_true, y_pred, i))

    if len(pk_values) == 0:
        return 0

    return sum(pk_values) / len(pk_values)


# another implementation of apk
# taken from:
# https://github.com/benhamner/Metrics/blob/
# master/Python/ml_metrics/average_precision.py

def ya_apk(actual, predicted, k=10):
    """
    yet another apk
    Computes the average precision at k.
    This function computes the AP at k between two lists of
    items.
    Parameters
    ----------
    actual : list
    A list of elements to be predicted (order doesn't matter)
    predicted : list
    A list of predicted elements (order does matter)
    k : int, optional
    The maximum number of predicted elements
    Returns
    -------
    score : double
    The average precision at k over the input lists
    """

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
         if p in actual and p not in predicted[:i]:
                 num_hits += 1.0
                 score += num_hits / (i+1.0)

    if not actual:
         return 0.0

    return score / min(len(actual), k)

def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k
    for a single sample
    Args:
        y_true: list of values, actual classes
        y_pred: list of values, predicated classes
    Results:
        mean avg precision at a given value k
    """
    apk_values = []

    for i in range(len(y_true)):
        apk_values.append(
            apk(y_true[i], y_pred[i], k=k)
        )

    return sum(apk_values)/len(apk_values)

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_log_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def mean_parcentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred)/ y_true)

def mean_abs_parcentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred)/ y_true)

def r2(y_true, y_pred):
    """
    This function calulates r-squared score
    Args:
        y_true: list of values, actual classes
        y_pred: list of values, predicated classes
    Results:
        r2 score
    """
    mean_true_value = np.mean(y_true)

    numerator = 0
    denominator = 0

    for yt, yp in zip(y_true, y_pred):
        numerator += (yt - yp) ** 2
        denominator += (yt - mean_true_value) ** 2

    ratio = numerator / denominator

    return  1 - ratio

def mcc(y_true, y_pred):
    """
    This function calculates Mattew's Correlation Coeffcient
    for binary classification
    Args:
        y_true: list of values, actual classes
        y_pred: list of values, predicated classes
    Results:
        mcc score
    """

    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_positive(y_true, y_pred)

    numerator = (tp * tn) - (fp * fn)
    denominator = (
        (tp + fp) * (fn + tn) *
        (fp + tn) * (tp + fn)
    )

    denominator = denominator ** 0.5

    return numerator/denominator
