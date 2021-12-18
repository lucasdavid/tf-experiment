from typing import Tuple
import tensorflow as tf


class Default:
  def augment(self, image):
    return image
  
  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)
  
  def call(self, image, label):
    image = self.augment(image)
    image = tf.cast(image, tf.float32)

    return image, label
  
  def augment_dataset(
      self,
      dataset: tf.data.Dataset,
      num_parallel_calls: int = None,
      as_numpy: bool = False,
      element_spec: Tuple[tf.TensorSpec] = None,
  ) -> tf.data.Dataset:
    if not as_numpy:
      return dataset.map(self.call, num_parallel_calls=num_parallel_calls)

    return dataset.map(
      lambda x, y: tf.py_function(self.call, inp=[x, y], Tout=element_spec),
      num_parallel_calls=num_parallel_calls
    )
