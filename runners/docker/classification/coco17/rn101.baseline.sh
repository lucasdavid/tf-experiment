#!/bin/bash

SERVICE=notebook
SOURCE=code/baseline.py
LOGS=./logs/classification/coco17/rn101.baseline
CONFIG=config/classification/coco17/rn101.baseline.yml


docker-compose exec $SERVICE python $SOURCE with $CONFIG -F $LOGS
