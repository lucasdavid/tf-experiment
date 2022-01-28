#@title Testing Loop

import os
import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import explain
from ..utils import logged, masked, normalize
from .metrics import (average_drop, average_drop_of_others,
                      increase_in_confidence)


@logged
def evaluate(
    nn: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_id: str,
    classes: List[str],
    methods: List[str],
    inference: Dict[str, Union[int, float]],
    ckpt_dir: str = '/tmp',
):
  nns = tf.keras.Model(
    inputs=nn.inputs,
    outputs=[nn.output, nn.get_layer('avg_pool').input],
    name=f'{nn.name}_ak')

  os.makedirs(ckpt_dir, exist_ok=True)

  return pd.concat(
    evaluate_one(
      nn,
      nns,
      dataset,
      vis_method,
      classes,
      inference,
      ckpt_file=os.path.join(ckpt_dir, f'{nn.name}_{dataset_id}_{vis_method}.ckpt')
    ).assign(method=vis_method)
    for vis_method in methods
  )


@logged
def evaluate_one(nn, nns, dataset, vis_method, classes, inference, ckpt_file):
  vis_method = explain.get(vis_method)
  print(f'Testing {vis_method.__name__}')

  cam_modifier = lambda c: normalize(tf.nn.relu(c))

  if os.path.exists(ckpt_file):
    with open(ckpt_file, 'rb') as f:
      context = pickle.load(f)
    steps = context['step'] + 1
    metrics = context['metrics']

    print(f'Checkpoint file {ckpt_file} found. {steps} steps skipped.')
    dataset = dataset.skip(steps)
  else:
    print(f'Checkpoint file {ckpt_file} not found. Starting from scratch.')
    steps = 0
    metrics = (np.zeros(len(classes), np.uint16),
               np.zeros(len(classes)),
               np.zeros(len(classes)),
               np.zeros(len(classes)),
               np.zeros(len(classes)),
               np.zeros(len(classes), np.uint16))

  return _evaluate_run(nn, nns, dataset, vis_method, cam_modifier, ckpt_file,
                       metrics, classes, inference, steps)


def _evaluate_run(
    nn,
    nns,
    dataset,
    cam_method,
    cam_modifier,
    ckpt_file,
    metrics,
    classes,
    inference,
    step=0,
):
  w = nn.layers[-1].weights[0]

  metric_names = ('increase %', 'avg drop %', 'avg retention %',
                  'avg drop of others %', 'avg retention of others %',
                  'detections')

  for x, bbox, label in dataset:
    label = tf.reduce_max(tf.one_hot(label, depth=len(classes)), axis=0)

    p, m = cam_method(nns, x, label, w)
    p = tf.nn.sigmoid(p)
    m = cam_modifier(m[..., tf.newaxis])
    
    results = cam_evaluation_step(nn, x, p, m, **inference)

    for e, f in zip(metrics, results):
      e += f

    with open(ckpt_file, 'wb') as f:
      pickle.dump({'step': step, 'metrics': metrics}, f)

    print('.', end='' if (step+1) % 80 else '\n')
    step += 1
  print()

  metrics, detections = metrics[:-1], metrics[-1]

  results = {n: m / detections for n, m in zip(metric_names, metrics)}
  results['label'] = classes
  results['detections'] = detections
  results = pd.DataFrame(results)

  print(f'Average Drop     %: {results["avg drop %"].mean():.2%}')
  print(f'Average Increase %: {results["increase %"].mean():.2%}')

  os.remove(ckpt_file)

  return results


def cam_evaluation_step(
    nn,
    x,
    p,
    m,
    batch_size: int = 32,
    threshold: float = explain.DEFAULT_THRESHOLD
):
  s = p > threshold
  w = tf.where(s)
  samples_ix, units_ix = w[:, 0], w[:, 1]

  md = tf.image.resize(m[s], x.shape[1:3])

  detections = tf.reduce_sum(tf.cast(s, tf.uint32), axis=0)

  y = p[s]                       # (batch, c)           --> (detections)
  xs = tf.gather(x, samples_ix)  # (batch, 300, 300, 3) --> (detections, 300, 300, 3)

  o = nn.predict(masked(xs, md), batch_size=batch_size)
  o = tf.nn.sigmoid(o)

  co = nn.predict(masked(xs, 1 -md), batch_size=batch_size)
  co = tf.nn.sigmoid(co)

  samples_ix, units_ix = samples_ix.numpy(), units_ix.numpy()

  incr = increase_in_confidence(p, y, o, samples_ix, units_ix)
  drop = average_drop(p, y, o, samples_ix, units_ix)
  rete = average_drop(p, y, co, samples_ix, units_ix)

  drop_of_others = average_drop_of_others(p, s, y, o, samples_ix, units_ix)
  rete_of_others = average_drop_of_others(p, s, y, co, samples_ix, units_ix)

  return incr, drop, rete, drop_of_others, rete_of_others, detections.numpy()
