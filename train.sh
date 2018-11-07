#!/bin/bash

python3 main.py --env_name "carla" --experiment-desc "carla_0_3_5_without/" \
    --save-model-dir "throttle=0.35, without randomization, with origin image, without smooth penalty" --max-training-num 1000000