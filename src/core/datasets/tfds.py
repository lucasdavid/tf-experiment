from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

from . import augment, tasks


def load(
    dataset_name,
    data_dir,
    splits=('train', 'test')
):
  print(f'Loading dataset {dataset_name} into {data_dir}')

  parts, info = tfds.load(
    dataset_name,
    split=splits,
    with_info=True,
    shuffle_files=True,
    data_dir=data_dir
  )
  return parts, info


def prepare(
    dataset: tf.data.Dataset,
    batch_size: int,
    sizes: Tuple[int],
    classes: int,
    task: str = 'classification',
    aug_policy: Union[bool, str, Callable] = False,
    aug_config: Dict[str, Any] = None,
    buffer_size: Union[int, str] = 'auto',
    parallel_calls: Union[int, str] = 'auto',
    preprocess_fn: Callable = None,
    drop_remainder: bool = True,
    randgen = None,
):
  if buffer_size == 'auto':
    buffer_size = tf.data.AUTOTUNE
  if parallel_calls == 'auto':
    parallel_calls = tf.data.AUTOTUNE

  task = partial(tasks.get(task), classes=classes, sizes=sizes)
  aug_policy = augment.get(aug_policy)
  aug_policy = partial(aug_policy, aug_config=aug_config, randgen=randgen, preprocess_fn=preprocess_fn)

  return (dataset
          .map(lambda entry: aug_policy(*task(entry)), num_parallel_calls=parallel_calls)
          .batch(batch_size, drop_remainder=drop_remainder)
          .prefetch(buffer_size))
