#!/bin/bash

SERVICE=notebook
SOURCE=src/baseline.py
LOGS=./logs/classification/coco17/rn50.randaug
CONFIG=config/classification/coco17/rn50.randaug.yml


docker-compose exec $SERVICE python $SOURCE with $CONFIG -F $LOGS
