#!/bin/bash

SERVICE=notebook
SOURCE=code/baseline.py
LOGS=./logs/classification/coco17/rn50.randaug
CONFIG=config/classification/coco17/rn50.randaug.yml


docker-compose exec $SERVICE python $SOURCE with $CONFIG \
  model.head.layer_class=kernel_usage \
  model.head.dropout_rate=0 \
  setup.paths.ckpt=./logs/classification/coco17/rn50.randaug/backup.ku/ \
  -F $LOGS_DIR
