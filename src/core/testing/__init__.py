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

from typing import Callable, List, Optional, Union

import pandas as pd
import tensorflow as tf

from ..inference import target_and_output
from . import explanations, tasks


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    classes: List[str],
    task: Union[str, Callable],
):
  print('-' * 32)
  print(f'Evaluation {str(task)}')

  outputs = target_and_output(model, dataset)
  evaluations = report(outputs, task, classes)

  return evaluations


def report(
    target_and_output,
    task: Union[str, Callable],
    classes: List[str] = None,
):
  task_fn = tasks.get(task)
  evaluations = task_fn(target_and_output, classes=classes)
  evaluations = pd.DataFrame(evaluations)
 
  print('-' * 32)
  print(str(task).replace('_', ' ').capitalize(), 'Report')
  with pd.option_context(
      'display.max_rows', None,
      'display.max_columns', None
  ):
    print(evaluations.round(4))

  return evaluations


__all__ = [
  'explanations',
  'evaluate',
  'report',
  'tasks',
]
