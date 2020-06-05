#!/bin/shF
PYTHONPATH=$(pwd):$PYTHONPATH python src_v2/train.py \
    --enc-backbone 50
