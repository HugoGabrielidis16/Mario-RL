#!/bin/bash

sudo docker pull yuuuugo/mario-rl
sudo docker run -it --gpus all -v $(pwd):/workspace light_mario-rl