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

from typing import Tuple

import tensorflow as tf

def normalize(
    x:          tf.Tensor,
    reduce_min: bool = True,
    reduce_max: bool = True,
    axis: Tuple[int] = (-3, -2)
):
  if reduce_min: x -= tf.reduce_min(x, axis=axis, keepdims=True)
  if reduce_max: x = tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))

  return x

def masked(
    x:    tf.Tensor,
    maps: tf.Tensor,
    axis: Tuple[int] = (-3, -2)
):
  return x * tf.image.resize(maps, [x.shape[i] for i in axis])
