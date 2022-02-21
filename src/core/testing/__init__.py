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

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import tensorflow as tf

from ..inference import target_and_output
from ..utils import logged
from . import tasks


@logged('Evaluation')
def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    classes: List[str],
    task: Union[str, Callable],
    kind: str = 'offline',
    config: Optional[Dict[str, Any]] = None,
):
  inputs = (target_and_output(model, dataset)
            if kind == 'offline'
            else (model, dataset))

  return report(task, inputs, config, classes)


def report(
    task: Union[str, Callable],
    inputs: Optional[List[Any]],
    config: Optional[Dict[str, Any]],
    classes: List[str] = None,
    verbose: int = 1,
):
  evaluations = tasks.get(task)(inputs, classes=classes, **(config or {}))
  evaluations = pd.DataFrame(evaluations).set_index('classes')

  scores = evaluations.select_dtypes('number')
  support = evaluations.support.values.reshape(-1, 1)

  evaluations.loc['avg_macro'] = scores.mean(axis=0)
  evaluations.loc['avg_weighted'] = (scores*support/support.sum()).sum(axis=0)

  if verbose:
    print('### Task Report', end='\n\n')
    print(evaluations.round(5).to_markdown())
    print()

  return evaluations


__all__ = [
  'evaluate',
  'report',
  'tasks',
]
