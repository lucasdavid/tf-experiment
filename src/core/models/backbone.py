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

from typing import Callable, Optional, Union

import tensorflow as tf


def get(
    input_tensor: tf.Tensor,
    architecture: Union[Callable, str],
    weights: Optional[str] = None,
    pooling: Optional[str] = None,
    trainable: bool = True,
):
  backbone_fn = (
    getattr(tf.keras.applications, architecture)
    if isinstance(architecture, str)
    else architecture
  )
  backbone = backbone_fn(
    input_tensor=input_tensor,
    weights=weights,
    pooling=pooling,
    include_top=False
  )

  if not trainable:
    backbone.trainable = False
  
  return backbone
