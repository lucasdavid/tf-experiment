# Copyright 2021 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import tensorflow as tf
from sklearn import metrics as skmetrics


def classification_multiclass(y, p, preds):
  try:
    roc_auc = skmetrics.roc_auc_score(y, p, average=None)
  except ValueError:
    roc_auc = None
  
  return {
    'auc_score': roc_auc,
    'confusion_matrix': skmetrics.confusion_matrix(y, preds).tolist()
  }


def classification_multilabel(y, p, predictions):
  try:
    roc_auc = skmetrics.roc_auc_score(y, p, average=None)
  except ValueError:
    roc_auc = None

  return {
    'auc_score': roc_auc,
    'confusion_matrix': skmetrics.multilabel_confusion_matrix(y, predictions).tolist()
  }


def classification_binary(y, p):
  tru_ = tf.reduce_sum(y, axis=0)
  neg_ = tf.reduce_sum(1 - y, axis=0)

  tpr = tf.reduce_sum(p * y, axis=0) / tru_
  fpr = tf.reduce_sum(p * (1 - y), axis=0) / neg_
  tnr = tf.reduce_sum((1 - p) * (1 - y), axis=0) / neg_
  fnr = tf.reduce_sum((1 - p) * y, axis=0) / tru_

  f2_score = skmetrics.fbeta_score(y, p, beta=2, average=None)
  precision, recall, f1_score, support = skmetrics.precision_recall_fscore_support(
      y, p, average=None
  )

  return {
    'tpr': tpr,
    'tnr': tnr,
    'fpr': fpr,
    'fnr': fnr,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'f2_score': f2_score,
    'support': support,
  }


def segmentation_multiclass(true_maps, predictions, probabilities):
  predicted_maps = np.argmax(predictions, axis=-1)

  true_maps == predicted_maps

  raise NotImplementedError


def detection_multilabel(bboxes, predictions, probabilities):
  raise NotImplementedError



def convert_to_corners(boxes):
  xyzw = (
    [boxes[..., :2] - boxes[..., 2:] / 2.0,
    boxes[..., :2] + boxes[..., 2:] / 2.0]
  )
  return np.concatenate(xyzw, axis=-1)


def iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
