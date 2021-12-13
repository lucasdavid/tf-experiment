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

def target_and_output(model, dataset):
  y_ = []
  o_ = []

  for batch, (x, t) in enumerate(dataset):
    o = model(x, training=False)

    y_.append(t)
    o_.append(o)

    print('.', end='' if (batch+1) % 120 else '\n')
  
  return (tf.concat(y_, axis=0),
          tf.concat(o_, axis=0))
