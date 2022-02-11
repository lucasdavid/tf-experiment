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

from functools import partial
from logging import warning
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ..utils import dig, get_preprocess_fn, to_list, unpack, logged
from . import augment, tasks, validate


def load(name, data_dir, splits=('train', 'test')):
  print(f'Loading dataset {name} into {data_dir}')

  parts, info = tfds.load(
      name, split=splits, with_info=True, shuffle_files=True, data_dir=data_dir
  )
  return splits, parts, info


def prepare(
    dataset: tf.data.Dataset,
    batch_size: int,
    sizes: Tuple[int],
    keys: Tuple[str],
    classes: int,
    split: str,
    take: Optional[int] = None,
    task: str = 'classification',
    validation: Dict[str, str] = None,
    augmentation: Optional[Dict[str, str]] = None,
    prefetch_buffer_size: Union[int, str] = 'auto',
    parallel_calls: Union[int, str] = 'auto',
    shuffle: Optional[Dict[str, Any]] = None,
    preprocess_fn: Callable = None,
    drop_remainder: bool = True,
    pad_values: float = -1,
    deterministic: bool = False
):
  ops = tf.data.Options()
  ops.deterministic = deterministic
  dataset = dataset.with_options(ops)

  if prefetch_buffer_size == 'auto':
    prefetch_buffer_size = tf.data.AUTOTUNE
  if parallel_calls == 'auto':
    parallel_calls = tf.data.AUTOTUNE

  if shuffle:
    print(f'  shuffling {split}: {shuffle}')
    shuffle = dict(**shuffle)
    shuffle_splits = shuffle.pop('splits', ['train'])
    if split in shuffle_splits:
      dataset = dataset.shuffle(**shuffle)

  # Validate entries, filtering out incoformin.
  v_key = dig(validation, 'key')
  v_kind = dig(validation, 'kind', default='not_empty')
  if v_key:
    dataset = dataset.filter(partial(validate.get(v_kind), key=v_key))

  # Extract specific task from entries.
  dataset = dataset.map(partial(tasks.get(task), classes=classes, sizes=sizes[:2], keys=keys))

  specs = dataset.element_spec
  shapes = tuple(s.shape for s in specs)

  # Build the augmentation policy.
  dont_aug_split = split not in dig(augmentation, 'splits', ())
  if not augmentation or dont_aug_split:
    augmentation = {'policy': 'Default'}

  over = augmentation.get('over', 'samples')
  aug_policy = augment.get(augmentation['policy'])

  pad_values = to_list(pad_values)
  if len(shapes) != 1 == len(pad_values):
    pad_values *= len(shapes)
  pad_values = [tf.convert_to_tensor(v, dtype=s.dtype) for v, s in zip(pad_values, specs)]

  aug_params = dict(
    over=over,
    element_spec=specs,
    num_parallel_calls=parallel_calls,
  )
  pad_params = dict(
    padded_shapes=shapes,
    drop_remainder=drop_remainder,
    padding_values=unpack(tuple(pad_values))
  )

  # (Maybe) augment dataset and batch samples.
  if over == 'samples':
    dataset = aug_policy.augment_dataset(dataset, **aug_params)
    dataset = dataset.padded_batch(batch_size, **pad_params)
  elif over == 'batches':
    dataset = dataset.padded_batch(batch_size, **pad_params)
    dataset = aug_policy.augment_dataset(dataset, **aug_params)
  else:
    raise ValueError(f'Illegal value "{over}" for augmentation.over config.')

  if take: dataset = dataset.take(take)

  if preprocess_fn:
    pfn = get_preprocess_fn(preprocess_fn)
    dataset = dataset.map(lambda x, *y: (pfn(tf.cast(x, tf.float32)), *y), parallel_calls)
  else:
    dataset = dataset.map(lambda x, *y: (tf.cast(x, tf.float32), *y), parallel_calls)

  if prefetch_buffer_size:
    dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset


@logged('Dataset Load and Prepare', with_arguments=False)
def load_and_prepare(
    config: Dict[str, Any],
):
  tfds.disable_progress_bar()

  splits, parts, info = load(**config['load'])
  print(f'{info.name}: {info.description}')

  if len(splits) < 3:
    raise ValueError('Three data.load.splits are expected (train, valid, test),'
                     f' but only {len(splits)} were passed: {splits}.')

  if len(splits) > 3:
    warning(f'  three data.load.splits are expected (train, valid, test). '
            f'These will be discarded: {splits[3:]}.')

  train, valid, test = (prepare(s, split=n, **config['prepare'])
                        for n, s in zip(splits, parts))

  print(f'  train: {train} ({splits[0]})')
  print(f'  valid: {valid} ({splits[1]})')
  print(f'  test:  {test}  ({splits[2]})')

  return train, valid, test, info


def classes(info) -> List[str]:
  labels_key = [k for k in info.features.keys() if k.startswith('label')]

  if not labels_key:
    raise ValueError(f'Cannot extract labels from {info}.')

  classes = info.features[labels_key[0]]

  if hasattr(classes, '_str2int'):
    classes = classes._str2int.keys()

  elif hasattr(classes, 'feature') and hasattr(classes.feature, '_str2int'):
    classes = classes.feature._str2int.keys()

  return np.asarray(list(classes))
