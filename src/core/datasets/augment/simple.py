from pprint import pprint
from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .default import Default


class Simple(Default):
  def __init__(
      self,
      constraints: Dict[str, Union[float, int]],
      random_generator: Union[tf.random.Generator, int] = None,
  ):
    if isinstance(random_generator, int):
      random_generator = tf.random.Generator.from_seed(random_generator, alg='philox')

    self.random_generator = random_generator
    self.constraints = constraints

  def augment(self, image):
    seeds = self.random_generator.make_seeds(7)

    image = tf.image.stateless_random_flip_left_right(image, seed=seeds[:, 0])
    image = tf.image.stateless_random_flip_up_down(image, seed=seeds[:, 1])
    image = tf.image.stateless_random_hue(image, self.constraints['hue_delta'], seed=seeds[:, 2])
    image = tf.image.stateless_random_brightness(image, self.constraints['brightness_delta'], seed=seeds[:, 3])
    image = tf.image.stateless_random_contrast(image, self.constraints['contrast_lower'], self.constraints['contrast_upper'], seed=seeds[:, 4])
    image = tf.image.stateless_random_saturation(image, self.constraints['saturation_lower'], self.constraints['saturation_upper'], seed=seeds[:, 5])

    angle = tf.random.stateless_uniform((), maxval=2*np.pi, seed=seeds[:, 6])
    image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')

    return image

  def augment_dataset(
      self,
      dataset: tf.data.Dataset,
      num_parallel_calls: int = None,
      over: str = 'samples',
      element_spec: Tuple[tf.TensorSpec] = None,
  ) -> tf.data.Dataset:
    print(f'  simple augment over dataset:')
    pprint(self.constraints)

    return dataset.map(
      lambda x, *y: (self.augment(x), *y),
      num_parallel_calls=num_parallel_calls)
