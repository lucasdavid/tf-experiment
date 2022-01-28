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
from ..utils import to_list, unpack


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
    take: Optional[int] = None,
    task: str = 'classification',
    augmentation: Optional[Dict[str, str]] = None,
    prefetch_buffer_size: Union[int, str] = 'auto',
    parallel_calls: Union[int, str] = 'auto',
    shuffle: Optional[Dict[str, Any]] = None,
    preprocess_fn: Callable = None,
    pad_drop_remainder: bool = True,
    pad_values: float = -1.0,
):
  if prefetch_buffer_size == 'auto':
    prefetch_buffer_size = tf.data.AUTOTUNE
  if parallel_calls == 'auto':
    parallel_calls = tf.data.AUTOTUNE

  if shuffle:
    ds = dataset.shuffle(**shuffle)

  ds = dataset.map(partial(tasks.get(task), classes=classes, sizes=sizes[:2], keys=keys))

  shapes = tuple(s.shape for s in ds.element_spec)
  over = augmentation.get('over', 'samples')
  aug_policy = augment.get(augmentation['policy'])

  aug_params = dict(
    over=over,
    element_spec=ds.element_spec,
    num_parallel_calls=parallel_calls,
  )
  pad_params = dict(
    padded_shapes=shapes,
    drop_remainder=pad_drop_remainder,
    padding_values=unpack(tuple(to_list(pad_values)))
  )

  if over == 'samples':
    ds = aug_policy.augment_dataset(ds, **aug_params)
    ds = ds.padded_batch(batch_size, **pad_params)
  elif over == 'batches':
    ds = ds.padded_batch(batch_size, **pad_params)
    ds = aug_policy.augment_dataset(ds, **aug_params)
  else:
    raise ValueError(f'Illegal value "{over}" for augmentation.over config.')

  if take: ds = ds.take(take)
  if preprocess_fn:
    preprocess_fn = utils.get_preprocess_fn(preprocess_fn)
    ds = ds.map(lambda x, *y: (preprocess_fn(x), *y), num_parallel_calls=parallel_calls)

  return ds.prefetch(prefetch_buffer_size)


def load_and_prepare(
    config: Dict[str, Any],
):
  tfds.disable_progress_bar()

  splits, info = load(**config['load'])
  print(f'{info.name}: {info.description}')
  
  train, valid, test = (prepare(s, **config['prepare']) for s in splits)
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
