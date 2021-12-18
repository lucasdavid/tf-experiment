from typing import Tuple
import tensorflow as tf


class Default:
  def augment(self, image):
    return image
  
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
      lambda x, y: (
        tf.ensure_shape(tf.py_function(self.augment, inp=[x], Tout=element_spec[0]),
                        element_spec[0].shape),
        y),
      num_parallel_calls=num_parallel_calls)
