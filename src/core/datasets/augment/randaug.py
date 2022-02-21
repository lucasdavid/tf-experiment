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

from typing import Tuple

import imgaug as ia
import tensorflow as tf
from imgaug import augmenters as iaa

rand_aug: iaa.RandAugment = None

from .default import Default


class RandAug(Default):
  def __init__(
      self,
      n: int = 3,
      m: int = 7,
      seed: int = 10482,
  ):
    self.rand_aug = iaa.RandAugment(n=n, m=m)

    ia.seed(seed)

  def augment(self, image):
    image = self.rand_aug(images=image.numpy())

    return image
  
  def augment_dataset(
      self,
      dataset: tf.data.Dataset,
      num_parallel_calls: int = None,
      over: str = 'samples',
      element_spec: Tuple[tf.TensorSpec] = None,
  ) -> tf.data.Dataset:
    print(f'  rand augmenting dataset')

    out_dtype = element_spec[0].dtype
    out_shape = element_spec[0].shape
    
    if over == 'batches':
      out_shape = [None, *out_shape]

    return dataset.map(
      lambda x, *y: (tf.ensure_shape(tf.py_function(self.augment, inp=[x], Tout=out_dtype), out_shape), *y),
      num_parallel_calls=num_parallel_calls)
