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

from typing import List, Tuple

import tensorflow as tf
from keras.utils.generic_utils import (
    deserialize_keras_object, serialize_keras_object
)

from . import metrics


def sparse_classification_multiclass(
    target_and_output: Tuple[tf.Tensor, tf.Tensor],
    threshold: float = 0.5,
    classes: List[str] = None,
):
  labels, probabilities = target_and_output
  target_and_output = (tf.one_hot(labels, depth=len(classes)), probabilities)

  return classification_multiclass(target_and_output, threshold, classes)


def classification_multiclass(
    target_and_output: Tuple[tf.Tensor, tf.Tensor],
    threshold: float = 0.5,
    classes: List[str] = None,
):
  labels, probabilities = target_and_output
  
  labels = labels.numpy()
  predictions = tf.argmax(probabilities, axis=1)
  
  return {
    **metrics.classification_binary(labels, tf.one_hot(predictions, depth=len(classes)).numpy()),
    **metrics.classification_multiclass(labels.argmax(axis=1), probabilities.numpy(), predictions.numpy())
  }


def classification_multilabel(
    target_and_output: Tuple[tf.Tensor, tf.Tensor],
    threshold: float = 0.5,
    classes: List[str] = None,
):
  labels, probabilities = target_and_output
  predictions = tf.cast(probabilities > threshold, probabilities.dtype).numpy()
  labels = labels.numpy()

  return {
    **metrics.classification_binary(labels, predictions),
    **metrics.classification_multilabel(labels, probabilities, predictions)
  }


def segmentation(
    target_and_output: Tuple[tf.Tensor, tf.Tensor],
    threshold: float = 0.5,
    classes: List[str] = None,
):
  maps, probabilities = target_and_output
  predictions = tf.cast(probabilities > threshold, probabilities.dtype).numpy()
  maps = maps.numpy()

  return metrics.segmentation_multiclass(maps, predictions, probabilities)


def serialize(metric):
  return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
    config,
    module_objects=globals(),
    custom_objects=custom_objects,
    printable_module_name='task evaluator function'
  )


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(f'Could not interpret task evaluator identifier: {identifier}')
