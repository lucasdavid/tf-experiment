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

from keras.utils.generic_utils import (
    deserialize_keras_object, serialize_keras_object
)

from .default import Default
from .simple import Simple
from .randaug import RandAug

def serialize(metric):
  return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
    config,
    module_objects=globals(),
    custom_objects=custom_objects,
    printable_module_name='augmentation function'
  )


def get(identifier) -> Default:
  if isinstance(identifier, bool):
    identifier = 'Simple' if identifier else 'Default'
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(f'Could not interpret augmentation identifier: {identifier}')

__all__ = [
  'Default',
  'Simple',
  'RandAug',
  'get',
]
