from typing import Any, Callable, Dict, Optional, Union
from .default import Default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Simple(Default):
  def __init__(
      self,
      random_generator: tf.random.Generator,
      constraints: Dict[str, Union[float, int]],
      preprocess_fn: Optional[Callable] = None,
  ):
    super().__init__(preprocess_fn)
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
