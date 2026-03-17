#!/bin/bash

sudo docker run -it --gpus all -v $(pwd):/workspace yuuuugo/light_mario-rl /bin/bash
