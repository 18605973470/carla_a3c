#!/bin/bash

python3 main.py --num-processes 3 --save-model-dir "carla_0_3_5_without/" \
    --experiment-desc "throttle=0.35, without randomization, with origin image, without smooth penalty" --max-training-num 1000000
