from typing import Callable, Optional, Union

import tensorflow as tf


def get(
    input_tensor: tf.Tensor,
    architecture: Union[Callable, str],
    weights: Optional[str] = None,
    pooling: Optional[str] = None,
    trainable: bool = True,
):
  backbone_fn = (
    getattr(tf.keras.applications, architecture)
    if isinstance(architecture, str)
    else architecture
  )
  backbone = backbone_fn(
    input_tensor=input_tensor,
    weights=weights,
    pooling=pooling,
    include_top=False
  )

  if not trainable:
    backbone.trainable = False
  
  return backbone
