import tensorflow as tf

def get(item):
  if isinstance(item, tf.keras.callbacks.Callback):
    return item
  
  if isinstance(item, tuple):
    cls, config = item
  elif isinstance(item, str):
    cls, config = item, {}
  elif isinstance(item, dict):
    cls, config = item['class_name'], item['config']
  else:
    raise ValueError(f'Cannot build a keras callback from the identifier {item}.')
  
  cls = (getattr(tf.keras.callbacks, cls, None)
         or getattr(tf.keras.callbacks.experimental, cls, None))

  return cls(**(config or {}))
