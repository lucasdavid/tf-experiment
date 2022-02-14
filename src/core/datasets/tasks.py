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

from logging import warning

from logging import warning

import tensorflow as tf
from keras.utils.generic_utils import (deserialize_keras_object,
                                       serialize_keras_object)

from ..utils import dig


def classification(entry, classes, sizes, keys):
  image, label = (dig(entry, k) for k in keys)

  if sizes is not None:
    image, _ = adjust_resolution(image, sizes)

  return tf.cast(image, tf.float32), label



def classification_multilabel_from_detection(entry, classes, sizes, keys):
  image, label = classification(entry, classes, sizes, keys)
  label = tf.reduce_max(tf.one_hot(label, depth=classes), axis=0)

  return image, label


def classification_multilabel_from_segmentation(entry, classes, sizes, keys):
  image, label = classification(entry, classes, sizes, keys)

  label = tf.reshape(label, [-1])
  label = tf.unique(label)[0]
  label = tf.reduce_max(tf.one_hot(label, depth=classes), axis=0)

  return image, label


def object_detection(entry, classes, sizes, keys):
  if not any('objects' in k for k in keys):
    warning(f'An object detection task is running, but "objects" is not in '
            f'keys={keys}. Make sure you are selecting the correct labels.')
  
  image, bboxes, label = (dig(entry, k) for k in keys)

  if sizes is not None:
    image, _ = adjust_resolution(image, sizes)
  
  return image, bboxes, label


def object_detection(entry, classes, sizes, keys):
  if not any('objects' in k for k in keys):
    warning(f'An object detection task is running, but "objects" is not in '
            f'keys={keys}. Make sure you are selecting the correct labels.')
  
  image, bboxes, label = (tf.cast(dig(entry, k), tf.float32) for k in keys)

  if sizes is not None:
    image, _ = adjust_resolution(image, sizes)

  return image, bboxes, label


def adjust_resolution(image, sizes):
  """Adjust input image sizes to be within the expected `sizes`.

  Example:
    shape = (512, 1024, 3)
    sizes = (512, 512)
    =>
    out shape = (256, 512, 3)

  """
  es = tf.constant(sizes, tf.float32)
  xs = tf.cast(tf.shape(image)[:2], tf.float32)

  ratio = tf.reduce_min(es / xs)
  xsn = tf.cast(tf.math.ceil(ratio * xs), tf.int32)

  image = tf.image.resize(image, xsn, preserve_aspect_ratio=True, method='nearest')
  image = tf.image.resize_with_crop_or_pad(image, *sizes)

  return image, ratio


def serialize(task):
  return serialize_keras_object(task)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='dataset task function'
  )


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(f'Could not interpret dataset task identifier: {identifier}')


__all__ = [
  'classification',
  'classification_multilabel_from_detection',
  'classification_multilabel_from_segmentation',
  'object_detection',
  'get',
]
