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
"""
Convert Model Checkpoint to Keras Model.

  Checkpoint → Keras saved model

"""

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from keras.distribute import worker_training_state

import core
from core.utils import dig

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(setup, dataset, model, training, evaluation, _log, _run):
  _log.info(__doc__)

  ex = core.experiment.setup({
    'setup': setup,
    'dataset': dataset,
    'model': model,
    'training': training,
    'evaluation': evaluation,
  }, _run, **setup)

  with ex.distributed_strategy.scope():
    model, backbone = core.models.classification.build_model(**model)
    core.models.summary(model)

  if dig(training, 'finetune.perform'):
    core.training.unfreeze_top_layers(backbone, **training['finetune']['unfreeze'])
  
  core.training.compile_distributed(
    model,
    loss=training['loss'],
    scale_loss=training['scale_loss'],
    optimizer=training['optimizer'],
    metrics=training['metrics'],
    distributed=ex.distributed_strategy,
  )

  model._training_state = (
    worker_training_state.WorkerTrainingState(model, ex.paths['ckpt']))
  model._training_state.restore()
  
  model.save(ex.paths['export'])

if __name__ == '__main__':
  ex.run_commandline()
