import tensorflow as tf
from keras.utils.generic_utils import (deserialize_keras_object,
                                       serialize_keras_object)

from . import constants
from .utils import normalize


@tf.function
def cam(model, x, y, w, threshold=constants.CLASSIFICATION_THRESHOLD):
  print(f'CAM tracing x:{x.shape} y:{y.shape}')

  l, a = model(x, training=False)
  maps = tf.einsum('bhwk,ku->buhw', a, w)

  return l, maps


@tf.function
def gradcampp(model, x, y, w, threshold=constants.CLASSIFICATION_THRESHOLD):
  print(f'Grad-CAM++ tracing x:{x.shape} y:{y.shape}')

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x)
    s, a = model(x, training=False)

  dsda = tape.batch_jacobian(s, a)

  dyda = tf.einsum('bu,buhwk->buhwk', tf.exp(s), dsda)
  d2 = dsda**2
  d3 = dsda**3
  aab = tf.reduce_sum(a, axis=(1, 2))               # --> (bk)
  akc = tf.math.divide_no_nan(
    d2,
    2.*d2 + tf.einsum('bk,buhwk->buhwk', aab, d3))  # --> (2*(buhwk) + bk*buhwk)

  weights = tf.einsum('buhwk,buhwk->buk', akc, tf.nn.relu(dyda))  # w: buk
  maps = tf.einsum('buk,bhwk->buhw', weights, a)                  # a:bhwk, m: buhw

  return s, maps


def scorecam(model, x, y, w, threshold=constants.CLASSIFICATION_THRESHOLD, acts_used=None):
  l, a = model(x, training=False)

  if acts_used == 'all' or acts_used is None:
    acts_used = a.shape[-1]

  # Sorting kernels from highest to lowest variance.
  std = tf.math.reduce_std(a, axis=(1, 2))
  a_high_std = tf.argsort(std, axis=-1, direction='DESCENDING')[:, :acts_used]
  a = tf.gather(a, a_high_std, axis=3, batch_dims=1)

  s = tf.Variable(tf.zeros((x.shape[0], y.shape[1], *x.shape[1:3])), name='sc_maps')

  for k in range(acts_used):
    ak = a[..., k:k+1]
    if tf.reduce_min(ak) == tf.reduce_max(ak): break
    s.assign_add(_scorecam_feed(x, ak))

  return l, s

@tf.function
def _scorecam_feed(model, x, ak, sizes):
  print(f'Score-CAM feed tracing x={x.shape} ak={ak.shape}, sizes={sizes}')
  ak = tf.image.resize(ak, sizes)

  b = normalize(ak)
  fm = model(x * b, training=False)
  fm = tf.nn.sigmoid(fm)
  fm = tf.einsum('bc,bhw->bchw', fm, ak[..., 0])

  return fm


@tf.function
def minmax_cam(model, x, y, w, threshold=constants.CLASSIFICATION_THRESHOLD):
  print(f'MinMax-CAM (tracing x:{x.shape} p:{y.shape})')

  l, a = model(x, training=False)
  p = tf.nn.sigmoid(l)

  d = tf.cast(p > threshold, tf.float32)
  c = tf.reduce_sum(d, axis=-1)
  c = tf.reshape(c, (-1, 1, 1))

  w = d[:, tf.newaxis, :] * w[tf.newaxis, ...]
  w_n = tf.reduce_sum(w, axis=-1, keepdims=True)
  w_n = w_n - w

  w = w - w_n / tf.maximum(c-1, 1)

  maps = tf.einsum('bhwk,bku->buhw', a, w)
  return l, maps


@tf.function
def d_minmax_cam(model, x, y, w, threshold=constants.CLASSIFICATION_THRESHOLD):
  print(f'D-MinMax-CAM (tracing x:{x.shape} y={y.shape})')

  l, a = model(x, training=False)
  p = tf.nn.sigmoid(l)

  d = tf.cast(p > threshold, tf.float32)
  c = tf.reshape(tf.reduce_sum(d, axis=-1), (-1, 1, 1))

  w = d[:, tf.newaxis, :] * w[tf.newaxis, ...]
  wa = tf.reduce_sum(w, axis=-1, keepdims=True)
  wn = wa - w

  w = (  tf.nn.relu(w)
       - tf.nn.relu(wn) / tf.maximum(c-1, 1)
       + tf.minimum(0., wa) / tf.maximum(c, 1))

  maps = tf.einsum('bhwk,bku->buhw', a, w)
  return l, maps


def serialize(explaining_func):
  return serialize_keras_object(explaining_func)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='explaining function'
  )


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(f'Could not interpret explaining function identifier: {identifier}')
