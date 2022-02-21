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
# ==============================================================================

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
