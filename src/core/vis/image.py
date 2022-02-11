# Copyright 2021 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from math import ceil
from typing import Callable, Optional

import tensorflow as tf


def save_image_samples(
    ds: tf.data.Dataset,
    to_file: str = 'samples.jpg',
    preprocess_fn: Optional[Callable] = None,
    raises_on_error: bool = False,
):
  try:
    (x, y), = ds.take(1).as_numpy_iterator()
    visualize((127.5*(x+1.)).astype('uint8'), rows=4, figsize=(20, 4), to_file=to_file)

    del x, y

  except Exception as ex:
    if raises_on_error:
      raise ex
    print(f'failed to save image samples: {len(ex)}', sys.stderr)


def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    figsize=(16, 7.2),
    cmap=None,
    to_file=None
):
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.set_style("whitegrid", {'axes.grid': False})

  if image is not None:
    if isinstance(image, (list, tuple)) or len(image.shape) > 3:  # many images
      plt.figure(figsize=figsize)
      cols = cols or ceil(len(image) / rows)
      for ix in range(len(image)):
        plt.subplot(rows, cols, ix + 1)
        visualize(
            image[ix],
            cmap=cmap,
            title=title[ix] if title is not None and len(title) > ix else None)
      
      plt.tight_layout()
      plt.subplots_adjust(wspace=0, hspace=0)

      if to_file is not None:
        print('  saving graphics to', to_file)
        plt.savefig(to_file)

      return

    if isinstance(image, tf.Tensor):
      image = image.numpy()
    if image.shape[-1] == 1:
      image = image[..., 0]
    plt.imshow(image, cmap=cmap)

  if title is not None:
    plt.title(title)
  plt.axis('off')
