from __future__ import division
from carla_wrapper import CarlaEnvironmentWrapper

def create_env(args, port, render, rank):
    # width, height
    env = CarlaEnvironmentWrapper(1, 84, 84, randomization=False,
                                      control_onput=1, port=port, rank=rank, preprocess="origin", render=render)
    return env
