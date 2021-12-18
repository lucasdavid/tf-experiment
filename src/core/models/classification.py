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

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Dropout


def head(
    input_tensor: tf.Tensor,
    backbone: tf.keras.Model,
    units: int,
    activation: Optional[str] = 'linear',
    dropout_rate: Optional[float] = None,
    name: str = None,
):
  y = backbone(input_tensor)
  y = Dropout(rate=dropout_rate, name='head/drop')(y)
  y = Dense(units, name='head/logits')(y)
  y = Activation(activation, dtype='float32', name='head/predictions')(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs=y,
    name=name
  )
