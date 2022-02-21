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

from keras.utils.layer_utils import count_params

from . import classification

def summary(model, print_fn=print):
  print_fn(f'Model {model.name}:')
  print_fn(' →  '.join(f'{l.name} ({type(l).__name__})' for l in model.layers))

  trainable_params = count_params(model.trainable_weights)
  non_trainable_params = count_params(model.non_trainable_weights)
  
  print_fn(f'Total params:     {trainable_params + non_trainable_params}')
  print_fn(f'Trainable params: {trainable_params}')


__all__ = [
  # 'backbones',
  'classification',
  'summary'
]
