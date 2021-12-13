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

import numpy as np
import tensorflow as tf


def plot_occurrence_and_prediction_matrices(
    l,
    p,
    threshold=0.5,
    classes=None,
    to_file='occur_pred.png'
):
  import matplotlib.pyplot as plt
  import seaborn as sns

  co_occurrence = tf.transpose(l) @ l
  co_occurrence_rate = tf.math.divide_no_nan(
    co_occurrence,
    tf.reshape(np.diag(co_occurrence), (-1, 1)))

  d = tf.cast(p > threshold, p.dtype)
  co_prediction = tf.transpose(d) @ d
  co_prediction_rate = tf.math.divide_no_nan(
    co_prediction,
    tf.reshape(np.diag(co_prediction), (-1, 1)))

  plt.figure(figsize=(24, 12))
  plt.subplot(131)
  plt.title('Co-occurrence Rates')
  sns.heatmap(
    co_occurrence_rate.numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )
  
  plt.subplot(132)
  plt.title('Co-prediction Rates')
  sns.heatmap(
    co_prediction_rate.numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )
  
  plt.subplot(133)
  plt.title('Difference')
  sns.heatmap(
    tf.abs(co_occurrence_rate - co_prediction_rate).numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )

  plt.tight_layout()

  if to_file:
    plt.savefig(to_file);
