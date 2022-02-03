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
python src/baseline.py with config/classification.cifar10.yml \
  model.backbone.architecture=ResNet101V2                     \
  -F logs/cifar10/baseline/
```
### Docker

The simplest way to run an experiment is to start the container and run the python interpreter
inside the container, which can be achieved by prepending the previous run command with
`docker-compose exec notebook`. For example:
```shell
SOURCE=src/baseline.py
CONFIG=config/classification/cifar10/rn50.baseline.yml
LOGS=logs/classification/cifar10/rn50.baseline

docker-compose up -d
docker-compose exec notebook python $SOURCE with $CONFIG -F $LOGS
```

You can change the parameters of the run by modifying the `CONFIG` file or by appending
the desired configuration after `with`:

```shell
docker-compose exec notebook python $SOURCE with $CONFIG \
  dataset.prepare.preprocess_fn=keras.applications.vgg.preprocess_input
  model.backbone.architecture=VGG16 \
  -F $LOGS
```

Check the runners at [runners/docker](/runners/docker) for examples on how to run
experiments in the docker environment.

### SDumont

Check the runners at [runners/sdumont](/runners/sdumont) for examples on how to run
experiments on the LNCC Santos Dumont Super Computer.
