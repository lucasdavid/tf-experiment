from keras.utils.layer_utils import count_params

from . import backbone, classification

def summary(model, print_fn=print):
  print_fn(f'Model {model.name}')
  print_fn(' →  '.join(f'{l.name} ({type(l).__name__})' for l in model.layers))

  trainable_params = count_params(model.trainable_weights)
  non_trainable_params = count_params(model.non_trainable_weights)
  
  print_fn(f'Total params:     {trainable_params + non_trainable_params}')
  print_fn(f'Trainable params: {trainable_params}')


__all__ = [
  'backbone',
  'classification',
  'summary'
]
