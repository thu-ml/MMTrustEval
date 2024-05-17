from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional, Union
import numpy as np
import scipy

"""
Input Requirement
y_true: 1d array-like
y_pred: 1d array-like
"""

def pred_no_op(y_true, y_pred):
    return y_pred

def pred_sum(y_true, y_pred):
    return np.array(y_pred).sum()

def pred_mean(y_true, y_pred):
    return np.array(y_pred).mean()

def pearson_corr(y_true, y_pred, nan_to_num: Optional[Union[float, int]] = None):
    x = np.array(y_pred, dtype=np.float32)
    if nan_to_num is not None:
        x = np.nan_to_num(x, nan=float(nan_to_num))
    y = np.array(y_true, dtype=np.float32)
    non_nan_indices = np.where(~np.isnan(x))[0]
    if non_nan_indices.size >= 2:
        corr = scipy.stats.pearsonr(x[non_nan_indices], y[non_nan_indices])[0]
    else:
        corr = np.nan
    return corr

def failure(y_true, y_pred, fails_num: Optional[Union[float, int]] = np.nan):
    # Calculate the proportion of occurrences of fails_num in the y_pred sequence.
    x = np.array(y_pred, dtype=np.float32)
    if np.isnan(fails_num):
        failure = np.isnan(x).sum() / x.size
    else:
        failure = (x == fails_num).sum() / x.size
    return failure

_supported_metrics = {
    # aggregation op
    "pred_no_op": pred_no_op,
    "pred_sum": pred_sum,
    "pred_mean": pred_mean,

    # general metrics
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score, 
    "f1_score": f1_score,
    "pearson_corr": pearson_corr,
    "failure": failure,
}
