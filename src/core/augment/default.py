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
      over: str = 'samples',
      element_spec: Tuple[tf.TensorSpec] = None,
  ) -> tf.data.Dataset:
    if not as_numpy:
      return dataset.map(
        lambda x, y: (self.augment(x), y),
        num_parallel_calls=num_parallel_calls)
    
    out_dtype = element_spec[0].dtype
    out_shape = element_spec[0].shape
    if over == 'batches':
      out_shape = [None, *out_shape]

    return dataset.map(
      lambda x, y: (tf.ensure_shape(tf.py_function(self.augment, inp=[x], Tout=out_dtype), out_shape), y),
      num_parallel_calls=num_parallel_calls)
