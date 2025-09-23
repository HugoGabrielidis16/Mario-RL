#!/bin/bash


COMMAND="python3 train.py \
    --batch-size 64 \
    --frame-stack 4 \
    --max-steps 5000  \
    --episodes 5000 \
    --test-every 50 \
    --moveset balanced"

docker pull yuuuugo/mario-rl
docker run -it --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"
