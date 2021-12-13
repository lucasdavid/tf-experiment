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

import tensorflow as tf

def classification(entry, classes, sizes):
  image, label = entry['image'], entry['label']

  if sizes is not None:
    image, _ = adjust_resolution(image, sizes)

  return image, label

def classification_multilabel_from_detection(entry, classes, sizes):
  image, label = entry['image'], entry['objects']['label']

  label = tf.reduce_max(tf.one_hot(label, depth=classes), axis=0)

  if sizes is not None:
    image, _ = adjust_resolution(image, sizes)

  return image, label

def adjust_resolution(image, sizes):
  es = tf.constant(sizes, tf.float32)
  xs = tf.cast(tf.shape(image)[:2], tf.float32)

  ratio = tf.reduce_min(es / xs)
  xsn = tf.cast(tf.math.ceil(ratio * xs), tf.int32)

  image = tf.image.resize(image, xsn, preserve_aspect_ratio=True, method='nearest')
  image = tf.image.resize_with_crop_or_pad(image, *sizes)

  return image, ratio


def get(identity):
  return globals()[identity]

__all__ = [
  'classification',
  'classification_multilabel_from_detection',
  'get',
]
