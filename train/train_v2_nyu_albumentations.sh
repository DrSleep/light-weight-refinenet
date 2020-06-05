#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src_v2/train.py \
    --enc-backbone 50 \
    --augmentations-type "albumentations"
