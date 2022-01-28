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
from .utils import to_list, unpack


def target_and_output(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    verbose: int = 1,
):
  targets = []
  outputs = []

  for step, (x, *t) in enumerate(dataset):
    o = model(x, training=False)

    targets.append(t)
    outputs.append(to_list(o))

    if verbose > 0:
      print('.', end='' if (step+1) % 120 else '\n')

  return (unpack([tf.concat(t, axis=0) for t in zip(*targets)]),
          unpack([tf.concat(o, axis=0) for o in zip(*outputs)]))
