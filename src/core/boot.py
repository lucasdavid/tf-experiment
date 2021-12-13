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


def gpus_with_memory_growth():
  gpus = list(tf.config.list_physical_devices('GPU'))

  print(f'Number of devices: {len(gpus)}')

  for d in gpus:
    print(d)
    print(f'  Setting device {d} to memory-growth mode.')
    
    try:
      tf.config.experimental.set_memory_growth(d, True)
    except Exception as e:
      print(e)


def appropriate_distributed_strategy():
  if tf.config.list_physical_devices('GPU'):
    return tf.distribute.MirroredStrategy()
  else:
    return tf.distribute.get_strategy()
