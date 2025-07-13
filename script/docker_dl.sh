#!/bin/bash

docker pull yuuuugo/mario-rl
docker run -it --gpus all -v $(pwd):/workspace mario-rl