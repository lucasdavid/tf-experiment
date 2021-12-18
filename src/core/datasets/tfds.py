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
    take: Optional[int] = None,
    task: str = 'classification',
    augmentation: Optional[Dict[str, str]] = None,
    buffer_size: Union[int, str] = 'auto',
    parallel_calls: Union[int, str] = 'auto',
    preprocess_fn: Callable = None,
    drop_remainder: bool = True,
):
  if buffer_size == 'auto':
    buffer_size = tf.data.AUTOTUNE
  if parallel_calls == 'auto':
    parallel_calls = tf.data.AUTOTUNE

  ds = dataset.map(partial(tasks.get(task), classes=classes, sizes=sizes[:2], keys=keys),
                   num_parallel_calls=parallel_calls)
  
  shapes = tuple(s.shape for s in ds.element_spec)
  aug_over = augmentation.get('over', 'samples')
  aug_policy = augment.get(augmentation['policy'])
  aug_parameters = dict(
    num_parallel_calls=parallel_calls,
    as_numpy=augmentation.get('as_numpy'),
    element_spec=ds.element_spec
  )

  if aug_over == 'samples':
    ds = aug_policy.augment_dataset(ds, **aug_parameters)
    ds = ds.padded_batch(batch_size, padded_shapes=shapes, drop_remainder=drop_remainder)
  elif aug_over == 'batches':
    ds = ds.padded_batch(batch_size, padded_shapes=shapes, drop_remainder=drop_remainder)
    ds = aug_policy.augment_dataset(ds, **aug_parameters)
  else:
    raise ValueError(f'Illegal value "{aug_over}" for augmentation.over config.')

  if take: ds = ds.take(take)
  if preprocess_fn:
    preprocess_fn = utils.get_preprocess_fn(preprocess_fn)
    ds = ds.map(lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y), num_parallel_calls=parallel_calls)

  return ds.prefetch(buffer_size)


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


def infer_shapes_from_task(
  task: str,
  sizes: Tuple[int],
  classes: int,
  batch_size: int,
  drop_remainder: bool,
) -> Tuple[Tuple[int], Tuple[int]]:
  if not drop_remainder:
    batch_size = None

  in_s = [batch_size, *sizes]
  out_s = [batch_size]

  if task != 'classification':
    # Any dense or one-hot targete
    out_s += [classes]

  return (in_s, out_s)
