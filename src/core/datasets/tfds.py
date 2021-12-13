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

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import augment, tasks
from ..utils import get_preprocess_fn


def load(name, data_dir, splits=('train', 'test')):
  print(f'Loading dataset {name} into {data_dir}')

  parts, info = tfds.load(
      name, split=splits, with_info=True, shuffle_files=True, data_dir=data_dir
  )
  return parts, info


def prepare(
    dataset: tf.data.Dataset,
    batch_size: int,
    sizes: Tuple[int],
    classes: int,
    task: str = 'classification',
    augmentation: Optional[Dict[str, str]] = None,
    buffer_size: Union[int, str] = 'auto',
    parallel_calls: Union[int, str] = 'auto',
    preprocess_fn: Callable = None,
    drop_remainder: bool = True,
    randgen=None,
):
  if buffer_size == 'auto':
    buffer_size = tf.data.AUTOTUNE
  if parallel_calls == 'auto':
    parallel_calls = tf.data.AUTOTUNE

  aug_policy = augmentation['policy']
  aug_config = augmentation['config']

  task = partial(tasks.get(task), classes=classes, sizes=sizes)
  aug_policy = augment.get(aug_policy)
  aug_policy = partial(
    aug_policy,
    aug_config=aug_config,
    randgen=randgen,
    preprocess_fn=get_preprocess_fn(preprocess_fn)
  )

  return (
    dataset
    .map(lambda entry: aug_policy(*task(entry)), num_parallel_calls=parallel_calls)
    .batch(batch_size, drop_remainder=drop_remainder)
    .take(10)
    .prefetch(buffer_size)
  )


def load_and_prepare(
    config: Dict[str, Any],
    randgen: tf.random.Generator,
):
  parts, info = load(**config['load'])

  print(f'{info.name}: {info.description}')

  train, valid, test = (
      prepare(part, randgen=randgen, **config['prepare']) for part in parts
  )

  print(f'train: {train}')
  print(f'valid: {valid}')
  print(f'test:  {test}')

  return train, valid, test, info


def classes(info):
  labels_key = [k for k in info.features.keys() if k.startswith('label')]

  if not labels_key:
    raise ValueError(f'Cannot extract labels from {info}.')

  classes = info.features[labels_key[0]]

  if hasattr(classes, '_str2int'):
    classes = classes._str2int.keys()

  elif hasattr(classes, 'feature') and hasattr(classes.feature, '_str2int'):
    classes = classes.feature._str2int.keys()

  return np.asarray(list(classes))
