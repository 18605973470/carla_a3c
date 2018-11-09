#!/bin/bash

python3 main.py  --experiment-desc "throttle=0.35, with start randomization, with origin image, with smooth penalty" \
    --save-model-dir "carla_0_3_5_with/" --num-processes 2 --num-steps 20 --max-training-num 8000000
