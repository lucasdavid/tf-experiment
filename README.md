# Tensorflow Experiment

A basic benchmark for tensorflow experiments.

## Setup
You can use this project in multiple distinct environments.
I list below a few configurations necessary for each one of these.

### Local
You are expected to set every single aspect of your local environment:
from GPU drivers to python package dependencies.

When using GPUs, you must have all of the Tensorflow's [software requirements](https://www.tensorflow.org/install/gpu#software_requirements)
set up. Mainly NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/cudnn).

You must also install the python dependencies appropriately:

```shell
pip install -r requirements.txt
```

### Docker
You can circumvent most configuration by using [docker](https://www.docker.com/),
[docker-compose](https://docs.docker.com/compose/) and
[docker-nvidia-container](https://github.com/nvidia/nvidia-container-runtime) (only if you're using a GPU).

To setup the entire infrastructure, simply type:
```shell
docker-compose build
```

If you don't have a GPU setup, and haven't installed the docker-nvidia-container,
remove the `runtime: nvidia` entry from the docker-compose.yml file,
and change the `BASE_IMAGE` to `tensorflow/tensorflow:latest-jupyter`.
Now try running the build and you should see Docker setting everything for you.

### SDumont Supercomputer

The job submitted to the executing queue must set the right environment.

For example, your job script can be:

```shell
#!/bin/bash
#...

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

pip3.9 install -r requirements.txt

python ...
```

## Running

### Local
```shell
SOURCE=src/train_and_finetune.py
LOGS=./logs/classification/cifar10/train.rn50.noaug

python $SOURCE                                                        \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  setup.paths.ckpt=$LOGS/backup                                       \
  -F $LOGS
```
### Docker

The simplest way to run an experiment is to start the container and run the python interpreter
inside the container, which can be achieved by prepending the previous run command with
`docker-compose exec notebook`. For example:
```shell
source config/docker/.env

SOURCE=src/train_and_finetune.py
LOGS=./logs/classification/cifar10/train.rn50.noaug


docker-compose exec $SERVICE python $SOURCE                           \
  with config/runs/classification/train_and_finetune.yml              \
  setup.paths.ckpt=$LOGS/backup                                       \
  -F $LOGS
```

You can add new mixins to modify components used in the experiment by
simply appending them to the command:

```shell
EXPERIMENT=voc12-noaug
EXPERIMENT_TAGS="['voc12', 'rn50']"

docker-compose exec $SERVICE python $SOURCE                           \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/classification/mixins/datasets/voc12.yml                \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR                                     \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS
```

Check the runners at [runners/docker](/runners/docker) for examples on how to run
experiments in the docker environment.

### SDumont

Check the runners at [runners/sdumont](/runners/sdumont) for examples on how to run
experiments on the LNCC Santos Dumont Super Computer.
