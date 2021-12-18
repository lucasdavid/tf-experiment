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
      output_shapes = None,
  ) -> tf.data.Dataset:
    if not as_numpy:
      return dataset.map(self.call, num_parallel_calls=num_parallel_calls)
    
    def augment_as_numpy(x, y):
      x, y = tf.py_function(self.call, inp=[x, y], Tout=[tf.float32, tf.float32])
      x = tf.ensure_shape(x, output_shapes[0])
      y = tf.ensure_shape(y, output_shapes[1])

      return x, y

    return dataset.map(augment_as_numpy, num_parallel_calls=num_parallel_calls)
