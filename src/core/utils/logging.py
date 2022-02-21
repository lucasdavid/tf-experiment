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

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


def printargs(
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    indent: int = 2
):
  ip = ' ' * indent
  np = ' ' * (indent+2)

  if args:
    print(f'{ip}args:', *map(str, args), sep=f'\n{np}')

  if kwargs:
    max_param_size = max(list(map(len, kwargs.keys())) or [0])

    for k, v in kwargs.items():
        print(f'{np}{k:<{max_param_size}} = {v}')


def log_begin(
    fun_name: str,
    *args,
    with_margins: bool = True,
    with_arguments: bool = True,
    heading: int = 2,
    indent: int = 0,
    **kwargs
):
  now = datetime.now()

  ip = ' '*indent
  hp = '#'*heading

  print()
  print(f'{ip}{hp} {fun_name + " ":—<65} {now.strftime("%H:%M:%S")}' if with_margins else fun_name)

  if with_arguments:
    printargs(args, kwargs, indent=indent)
    print()

  return now


def logged(
    name: Optional[str] = None,
    with_margins: bool = True,
    with_arguments: bool = True,
    heading: int = 2
) -> Callable:
  """Logged Wrapper.

  Wraps a function and log its call whenever happens, as well as its parameters and call time.

  Parameters
  ----------
  name: str
    Verbose (human) name of the section being logged.
    If nothing is passed, the name of the function is used instead.
  
  with_margins: bool
    Whether or not to add margins to the logged message.
    Set it to false to produce more compact reports.
  
  with_arguments: bool
    Whether or not to include the arguments of the function in the logged message.
    Set it to false to produce more compact reports.
  
  heading: int, default=2

  Examples
  --------

  >>> @logged('Model Training')
  >>> def training(model, dataset):
  >>>   model.fit(dataset, epochs=10)
  >>>   return model.history.history

  """
  def decorator(fn):
    def _log_wrapper(*args, **kwargs):
      log_begin(
        name or fn.__name__.replace('_', ' '),
        *args,
        with_margins=with_margins,
        with_arguments=with_arguments,
        heading=heading,
        **kwargs
      )
      return fn(*args, **kwargs)
    return _log_wrapper
  return decorator
