import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from core.utils import checkpoint as checkpoint_util
from core.utils import dig, masked, normalize
from core.utils.logging import logged

from .. import cams
from . import metrics
from .utils import bboxes_to_segmentation_label

METRIC_NAMES = ('IC%', 'AD%', 'AR%', 'ADO%', 'ARO%')


def cam_benchmark(
    inputs: Tuple[tf.keras.Model, tf.data.Dataset],
    method: str,
    ckpt_dir: str,
    inference: Optional[Dict[str, any]] = None,
    classes: List[str] = None,
    dataset_id: str = 'test',
    pooling: str = 'avg_pool',
):
  nn, dataset = inputs

  pl = nn
  for name in pooling.split('.'):
    pl = pl.get_layer(name)

  nns = tf.keras.Model(
    inputs=nn.inputs,
    outputs=[nn.output, pl.input],
    name=f'{nn.name}_ak')

  clf_layer = nn.get_layer('head/logits')
  clf_weights = getattr(clf_layer, 'regularized_kernel', clf_layer.kernel)

  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt = os.path.join(ckpt_dir, f'{nn.name}_{dataset_id}_{method}.ckpt')

  evaluations = _online_benchmark(nn, nns, clf_weights, dataset, method, classes, inference, ckpt)
  evaluations.update({'classes': classes})

  return evaluations


def _online_benchmark(
    nn, nns, w, dataset,
    vis_method, classes, inference, ckpt,
    activation: Optional[Callable] = None,
):
  iou_bbox_metrics = [f'IoU-Loc_{delta}' for delta in dig(inference, 'iou_bbox', ())]
  iou_metrics = [f'IoU_{delta}' for delta in dig(inference, 'iou', ())]
  metrics = [*METRIC_NAMES, *iou_bbox_metrics, *iou_metrics, 'occurrences', 'support']

  steps = 0
  scores = np.zeros((len(metrics), len(classes)))
  vis_method = cams.get(vis_method)
  cam_modifier = lambda c: normalize(tf.nn.relu(c))

  if os.path.exists(ckpt):
    steps, scores = checkpoint_util.restore(ckpt)
    steps += 1
    dataset = dataset.skip(steps)
    print(f'checkpoint file `{ckpt}` found. {steps} steps skipped.')

  for x, label, *other_labels in dataset:
    label_ohe = tf.one_hot(label, depth=len(classes))
    if label_ohe.shape.rank > 2:
      label_ohe = tf.reduce_max(label_ohe, axis=1)

    p, m = vis_method(nns, x, label_ohe, w)
    if activation:
      p = activation(p)
    m = cam_modifier(m[..., tf.newaxis])

    results = cam_evaluation_step(nn, x, label, label_ohe, other_labels, p, m, **inference)

    for e, f in zip(scores, results):
      e += f
    
    checkpoint_util.save(ckpt, steps, scores)

    print('.', end='' if (steps+1) % 80 else '\n')
    steps += 1
  print()

  support = scores[-1]
  results = {n: m / support for n, m in zip(metrics[:-2], scores)}
  results.update(support=support, occurrencess=scores[-2])

  os.remove(ckpt)

  return results


def cam_evaluation_step(
    nn,
    x,
    label,
    label_ohe,
    other_labels,
    p,
    m,
    batch_size: int = 32,
    threshold: float = 0.5,
    iou_bbox: List[float] = None,
    iou_panoptic: List[float] = None,
):
  s = p > threshold
  w = tf.where(s)
  samples_ix, units_ix = w[:, 0], w[:, 1]

  md = tf.image.resize(m[s], x.shape[1:3])
  occurrences = tf.reduce_sum(tf.cast(label_ohe, tf.uint32), axis=0)
  support = tf.reduce_sum(tf.cast(s, tf.uint32), axis=0)

  y = p[s]                       # (b, c)           --> (o)
  xs = tf.gather(x, samples_ix)  # (b, 300, 300, 3) --> (o, 300, 300, 3)

  o = nn.predict(masked(xs, md), batch_size=batch_size)
  co = nn.predict(masked(xs, 1 -md), batch_size=batch_size)

  samples_ix, units_ix = samples_ix.numpy(), units_ix.numpy()

  # computing metrics
  ic = metrics.ic(p, y, o, samples_ix, units_ix)
  
  ad = metrics.ad(p, y, o, samples_ix, units_ix)
  ar = metrics.ad(p, y, co, samples_ix, units_ix)

  ado = metrics.ado(p, s, y, o, samples_ix, units_ix)
  aro = metrics.ado(p, s, y, co, samples_ix, units_ix)

  scores = [ic, ad, ar, ado, aro]

  
  if iou_bbox:
    bbox = other_labels[0]
    shape = [*p.shape, *x.shape[1:3]]
    segs = tf.constant(bboxes_to_segmentation_label(label.numpy(), bbox.numpy(), shape))
    segs = segs[s]
    scores += [metrics.iou(p, segs, md[..., 0], samples_ix, units_ix, delta=d) for d in iou_bbox]
  
  if iou_panoptic:
    panoptic_image = other_labels[1]

    print(panoptic_image[0].numpy().astype('int'))
    print('panoptic_image', panoptic_image.shape, panoptic_image.dtype)
    print('md', md.shape, md.dtype)

    panoptic_image = tf.transpose(panoptic_image, [0, 3, 1, 2])[s] # (b,u,h,w) -> (d,h,w)
    print(panoptic_image.shape)

    scores += [metrics.iou(p, panoptic_image, md[..., 0], samples_ix, units_ix, delta=d) for d in iou_bbox]
  
  return scores + [occurrences.numpy(), support.numpy()]
