#!/bin/bash

SERVICE=notebook
SOURCE=code/baseline.py
LOGS=./logs/classification/coco17/rn50.randaug
CONFIG=config/classification/coco17/rn50.randaug.yml


docker-compose exec $SERVICE python $SOURCE with $CONFIG \
  model.head.kernel_initializer=orthogonal \
  model.head.kernel_regularizer=orthogonal \
  model.head.dropout_rate=0 \
  setup.paths.ckpt=./logs/classification/coco17/rn50.randaug/backup.orthogonal/ \
  -F $LOGS
