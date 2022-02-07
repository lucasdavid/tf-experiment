#!/bin/bash

SERVICE=notebook
SOURCE=code/baseline.py
LOGS=./logs/classification/coco17/rn101.randaug
CONFIG=config/classification/coco17/rn101.randaug.yml


docker-compose exec $SERVICE python $SOURCE with $CONFIG \
  model.head.dropout_rate=0 \
  training.perform=False \
  setup.paths.best=./logs/classification/coco17/rn101.randaug/4/best.h5 \
  -F $LOGS
