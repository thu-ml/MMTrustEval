from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional, Union
import numpy as np
import scipy
import json

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

def parse_box_string(box_str):
    box_str = box_str.replace("'''", "").strip("[]")
    parts = box_str.split(",")
    parsed_parts = []
    for part in parts:
        if '/' in part:
            numerator, denominator = part.split('/')
            parsed_parts.append(float(numerator) / float(denominator))
        else:
            parsed_parts.append(float(part))
    return parsed_parts

def iou_judge(box1_list, box2_list):
    print('box1_list: {}'.format(box1_list))
    print('box2_list: {}'.format(box2_list))
    cnt = 0
    box_len = len(box1_list)
    for i in range(box_len):
        box_1 = json.loads(box1_list[i])
        box_2 = parse_box_string(box2_list[i])
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)
        inter_area = inter_width * inter_height 
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if inter_area == 0 or union_area == 0:
            iou = 0
        else:
            iou = inter_area / union_area
        print("iou",iou)
        if iou > 0.5:
            print("success grounding!")
            cnt += 1
    
    grounding_rate = (cnt * 1.0 / box_len) * 100.0       
    return grounding_rate

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
    "iou_judge": iou_judge
}
