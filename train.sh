#!/bin/bash

python3 main.py --env_name "carla" --experiment-desc "throttle=0.35, without randomization, with origin image, without smooth penalty" \
    --save-model-dir "carla_0_3_5_without/" --num-steps 20 --max-training-num 1000000