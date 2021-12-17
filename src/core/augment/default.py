from typing import Callable, Optional

import tensorflow as tf


class Default:
  def __init__(
      self,
      preprocess_fn: Optional[Callable] = None
  ):
    self.preprocess_fn = preprocess_fn
  
  def augment(self, image):
    return image
  
  def __call__(self, image, label):
    image = self.augment(image)
    image = tf.cast(image, tf.float32)
    if self.preprocess_fn:
      image = self.preprocess_fn(image)

    return image, label
