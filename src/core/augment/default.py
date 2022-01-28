from typing import Tuple
import tensorflow as tf


class Default:
  def augment(self, image):
    return image
  
  def augment_dataset(
      self,
      dataset: tf.data.Dataset,
      num_parallel_calls: int = None,
      over: str = 'samples',
      element_spec: Tuple[tf.TensorSpec] = None,
  ) -> tf.data.Dataset:
    return dataset
