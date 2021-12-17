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

from . import tasks
from .. import augment, utils


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
    keys: Tuple[str],
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

  ds = dataset.map(partial(tasks.get(task), classes=classes, sizes=sizes, keys=keys),
                   num_parallel_calls=parallel_calls)

  augment.get(augmentation)

  aug_policy = augmentation.get('policy')
  aug_config = augmentation.get('config')
  aug_over = augmentation.get('over')
  aug_as_numpy = augmentation.get('as_numpy')

  aug_policy_filled = partial(
    augment.get(aug_policy),
    aug_config=aug_config,
    randgen=randgen,
    preprocess_fn=utils.get_preprocess_fn(preprocess_fn)
  )

  if aug_as_numpy:
    aug_policy_fn = lambda x, y: (tf.py_function(aug_policy_filled, [x, y], [tf.float32, tf.float32]))
  else:
    aug_policy_fn = aug_policy_filled

  if aug_over == 'samples':
    ds = (ds.map(aug_policy_fn, num_parallel_calls=parallel_calls)
            .batch(batch_size, drop_remainder=drop_remainder))
  elif aug_over == 'batches':
    ds = (ds.batch(batch_size, drop_remainder=drop_remainder)
            .map(aug_policy_fn, num_parallel_calls=parallel_calls))
  else:
    raise ValueError(f'Illegal value "{aug_over}" for augmentation.over config.')

  return ds.prefetch(buffer_size)


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

  if info.name == 'cityscapes':
    labels = ['road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge',
              'tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky',
              'person','rider','car','truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']

    return np.asarray(labels)

  if not labels_key:
    raise ValueError(f'Cannot extract labels from {info}.')

  classes = info.features[labels_key[0]]

  if hasattr(classes, '_str2int'):
    classes = classes._str2int.keys()

  elif hasattr(classes, 'feature') and hasattr(classes.feature, '_str2int'):
    classes = classes.feature._str2int.keys()

  return np.asarray(list(classes))
