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

from typing import Any, Callable, Dict, List, Union

import pandas as pd

from . import tasks


def report(
    target_and_output,
    task: Union[str, Callable],
    run_params: Dict[str, Any],
    report_path: str = None,
    classes: List[str] = None,
):
  task = tasks.get(task)
  evaluations = task(target_and_output, classes=classes)
  evaluations = pd.DataFrame(evaluations)

  evaluations.to_csv(report_path, index=False)
  
  print('-' * 32)
  print(task.replace('_', ' ').capitalize(), 'Report')
  print(evaluations.round(4))
