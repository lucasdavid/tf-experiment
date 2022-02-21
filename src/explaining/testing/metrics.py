import numpy as np
import tensorflow as tf


def ic(
    p,  # f(x)             (batch, classes)
    y,  # f(x)[f(x) > 0.5] (detections, 1)
    o,  # f(x*mask(x, m))  (detections, classes)
    samples_ix,
    units_ix
):
  oc = tf.gather(o, units_ix, axis=1, batch_dims=1)  # (detections, 1)

  incr = np.zeros(p.shape, np.uint32)
  incr[samples_ix, units_ix] = tf.cast(y < oc, tf.uint32).numpy()

  return incr.sum(axis=0)


def ad(p, y, o, samples_ix, units_ix):
  oc = tf.gather(o, units_ix, axis=1, batch_dims=1)

  drop = np.zeros(p.shape)
  drop[samples_ix, units_ix] = (tf.nn.relu(y - oc) / y).numpy()

  return drop.sum(axis=0)


def ado(p, s, y, o, samples_ix, units_ix):
  # Drop of all units, for all detections
  d = tf.gather(p, samples_ix)
  d = tf.nn.relu(d - o) / d

  # Remove drop of class `c` and non-detected classes
  detected = tf.cast(tf.gather(s, samples_ix), tf.float32)
  d = d*detected
  d = tf.reduce_sum(d, axis=-1) - tf.gather(d, units_ix, axis=1, batch_dims=1)
  c = tf.reduce_sum(detected, axis=-1)

  # Normalize by the number of peer labels for detection `c`
  d = d / tf.maximum(1., c -1)

  drop = np.zeros(p.shape)
  drop[samples_ix, units_ix] = d.numpy()

  return drop.sum(axis=0)


def _iou(y, p, axis=(-3, -2)):
  i = tf.reduce_sum(y * p, axis=axis)
  u = tf.reduce_sum(tf.clip_by_value(y + p, 0, 1), axis=axis)
  return i / u

def iou(
    p,           # f(x) -> (batch, classes)
    segs,        # y    -> (detections, H, W)
    masks,       # m    -> (detections, H, W)
    samples_ix,
    units_ix,
    delta=0.15,  # [0.15, 0.25, 0.5]
):
  _iou_loc = _iou(segs, tf.cast(masks > delta, tf.float32))
  _loc = np.zeros(p.shape, np.float32)
  _loc[samples_ix, units_ix] = _iou_loc

  return _loc.sum(axis=0)
