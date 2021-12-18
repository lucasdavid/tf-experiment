# Salient Segmentation

Tensorflow implementation of attention-and-saliency-based-segmentation.

## Running

Check the runners at [runners/sdumont](/runners/sdumont) for examples on how to run this 
at the LNCC Santos Dumont Super Computer.

```shell
python src/baseline.py with config/classification.cifar10.yml \
  model.backbone.architecture=ResNet101V2                     \
  -F logs/cifar10/baseline/
```
