import tensorflow as tf


class Orthogonal(tf.keras.regularizers.Regularizer):
  def __init__(self, l2=0.01):
    self.l2 = l2

  def __call__(self, w):
    units = w.shape[-1]

    if len(w.shape) > 2:
      w = tf.reshape(w, (-1, units))

    r = tf.matmul(w, w, transpose_a=True)
    r = r - tf.eye(units)

    return self.l2 * tf.reduce_mean(r**2)

  def get_config(self):
    return {'l2': self.l2}


def get(identifier):
  if not identifier:
    return identifier

  name = (str(identifier.get('class_name'))
          if isinstance(identifier, dict)
          else str(identifier))
  config = identifier.get('config', {}) if isinstance(identifier, dict) else {}

  if name.lower() == 'orthogonal':
    return Orthogonal.from_config(**config)

  return tf.keras.regularizers.get(identifier)
