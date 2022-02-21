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

from typing import Any, Dict, List, Optional, Union


def dig(
    o: Optional[Dict],
    prop: Union[str, List[str]],
    default: Any = None,
    required: bool = False
):
  if o is None:
    o = {}
  if isinstance(prop, str):
    prop = prop.split('.')

  for p in prop:
    if p in o or required:
      o = o[p]
    else:
      return default

  return o


def to_list(x):
  return x if isinstance(x, list) else [x]


def unpack(x):
  return x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x


def attempt(
    fn, description: str = None, indent: int = 0, verbose: int = 1, raise_errors: bool = False
):
  if verbose:
    if not description:
      description = fn.__name__.replace('_', ' ')

    ip = ' ' * indent
    print(f'{ip}{description}...', end=' ')

  try:
    output = fn()
  except Exception as e:
    if raise_errors:
      raise

    output = None
    if verbose:
      print(f'✘ {e}')
  else:
    if verbose:
      print('✔')

  return output
