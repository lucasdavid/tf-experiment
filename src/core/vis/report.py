
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
    plt.savefig(to_file)
