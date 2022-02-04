import tensorflow as tf
from keras.utils.generic_utils import (deserialize_keras_object,
                                       serialize_keras_object)

from ..utils import dig


def _value_of(entry, key):
  return dig(entry, key, required=True)


def not_empty(entry, key):
  value = _value_of(entry, key)
  print(f'validation rule: {key} not-empty value={value}')
  return tf.size(value) != 0


def is_positive(entry, key):
  value = _value_of(entry, key)
  print(f'validation rule: {key} is-positive value={value}')
  return tf.reduce_any(value >= 0)


def serialize(task):
  return serialize_keras_object(task)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='dataset validation function'
  )


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(f'Could not interpret dataset validation identifier: {identifier}')
