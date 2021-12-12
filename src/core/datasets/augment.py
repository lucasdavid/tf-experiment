import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def default_policy_fn(image, label, preprocess_fn, aug_config, randgen):
  image = tf.cast(image, tf.float32)

  return preprocess_fn(image), label


def augment_policy_fn(image, label, preprocess_fn, aug_config, randgen):
  seeds = randgen.make_seeds(7)

  image = tf.image.stateless_random_flip_left_right(image, seed=seeds[:, 0])
  image = tf.image.stateless_random_flip_up_down(image, seed=seeds[:, 1])
  image = tf.image.stateless_random_hue(image, aug_config['hue_delta'], seed=seeds[:, 2])
  image = tf.image.stateless_random_brightness(image, aug_config['brightness_delta'], seed=seeds[:, 3])
  image = tf.image.stateless_random_contrast(image, aug_config['contrast_lower'], aug_config['contrast_upper'], seed=seeds[:, 4])
  image = tf.image.stateless_random_saturation(image, aug_config['saturation_lower'], aug_config['saturation_upper'], seed=seeds[:, 5])

  angle = tf.random.stateless_uniform((), maxval=2*np.pi, seed=seeds[:, 6])
  image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
  
  image = tf.cast(image, tf.float32)

  return preprocess_fn(image), label


def get(policy):
  if isinstance(policy, bool):
    return augment_policy_fn if policy else default_policy_fn

  if isinstance(policy, str):
    return globals()[policy]

  return policy
