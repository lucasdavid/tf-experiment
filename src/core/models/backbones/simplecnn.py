from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers


def conv_block(
      x: tf.Tensor,
      filters: int,
      kernel_size=(3, 3),
      strides=(1, 1),
      batch_norm: bool = True,
      activation='relu',
      dropout: float = 0,
      padding: str = 'valid',
      name: str = None,
  ):
    if dropout:
      x = layers.Dropout(dropout, name=f'{name}/drop')(x)

    kwargs = dict(strides=strides, use_bias=not batch_norm, padding=padding, name=f'{name}/conv')
    x = layers.Conv2D(filters, kernel_size, **kwargs)(x)
    if batch_norm:
      x = layers.BatchNormalization(name=f'{name}/bn')(x)
    x = layers.Activation(activation, name=f'{name}/{activation}')(x)

    return x

def conv_stack(x, filters, name, block_activation, dropout, residual: bool = False):
  y = conv_block(x, filters, name=f'g{name}/b1', activation=block_activation, dropout=dropout)
  y = conv_block(y, filters, name=f'g{name}/b2', activation=block_activation, strides=2)

  shape = tf.shape(y)
  height, width = shape[1], shape[2]
  
  if residual:
    z = tf.image.resize(x, (height, width), name=f'g{name}/resize')
    x = layers.concatenate([z, y], name=f'g{name}/add')

  return x


def SimpleCNN(
    input_tensor: tf.Tensor,
    weights: Optional[str] = None,
    pooling: Optional[str] = None,
    stacks: int = 3,
    dropout: float = 0,
    block_activation: str = 'relu',
    include_top: bool = True,
    classes: int = 1000,
    activation: str = 'softmax',
):
  if weights:
    raise ValueError('This model does not have any checkpoints. You must set `weights=None`.')

  x = input_tensor
  x = conv_block(x, 32, padding='same', name='stack0/conv', activation=block_activation)

  for idx in range(stacks):
    x = conv_stack(x, 32*(idx+1), f'stack{idx+1}', block_activation, dropout, residual=True)
  
  if pooling:
    if pooling == 'avg':
      pool_layer = layers.GlobalAveragePooling2D(name='avg_pool')
    elif pooling == 'max':
      pool_layer = layers.GlobalMaxPool2D(name='max_pool')
    else:
      raise ValueError(f'Illegal value {pooling} for parameter `pooling`. Expected `avg` or `max`.')

    x = pool_layer(x)

    if include_top:
      x = layers.Dense(classes, name='predictions', activation=activation)(x)
  
  return tf.keras.Model(input_tensor, x, name=f'scnn{stacks}')
