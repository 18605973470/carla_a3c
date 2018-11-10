#!/usr/bin/python3

import os
import sys
from os import path, environ

os.environ["CARLA_ROOT"] = "/home/r720/Driving/Carla/"  # "/home/r720/Driving/Carla" #"/home/arch760/FanYang/Driving/"
# os.environ["SDL_HINT_CUDA_DEVICE"] = "0"
# os.environ["SDL_VIDEODRIVER"] = "offscreen"

try:
    if 'CARLA_ROOT' in environ:
        sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
    from carla.client import CarlaClient
    from carla import image_converter
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.sensor import Camera
    from carla.client import VehicleControl
except ImportError:
    print("can't find module 'carla' ")
    exit(0)
    # from logger import failed_imports
    # failed_imports.append("CARLA")

import cv2
import numpy as np
import gym
import logging
import subprocess
import signal

# 0: [0, 0, 0],  # None *
# 1: [70, 70, 70],  # Buildings
# 2: [190, 153, 153],  # Fences
# 3: [72, 0, 90],  # Other
# 4: [220, 20, 60],  # Pedestrians *
# 5: [153, 153, 153],  # Poles
# 6: [157, 234, 50],  # RoadLines *
# 7: [128, 64, 128],  # Roads *
# 8: [244, 35, 232],  # Sidewalks *
# 9: [107, 142, 35],  # Vegetation
# 10: [0, 0, 255],  # Vehicles
# 11: [102, 102, 156],  # Walls *
# 12: [220, 220, 0]  # TrafficSigns
import time


def map(v):
    return v
    # r = 0
    # if v == 0 or v == 3 or v == 5 or v > 10:
    #     r = 0
    # elif v == 6:
    #     r == 7
    # else:
    #     r = v
    # if v == 0 or v == 1 or v == 3 or v == 5 or v == 9 or v == 11 or v == 12 or v == 2 : # None
    #     r = 0
    # elif v == 8: # Sidewalks
    #     r = 8
    # elif v == 7:
    #     if np.random.rand() < 0.002:
    #         r = 0
    #     else:
    #         r = 7
    # elif v == 6 : # Roads
    #     if np.random.rand() < 0.45:
    #         r = 7
    #     else:
    #         r = 6
    # else:
    #     r = v # pedestrians Vehicles
    # if v == 7 or v == 6:
    #     r = 7
    # return r


def map_process(img):
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][j] = map(img[i][j])
    return result


def to_color(img):
    classes = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 255],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]  # TrafficSigns
    }
    result = np.zeros((img.shape[0], img.shape[1], 3))
    for key, value in classes.items():
        result[np.where(img == key)] = value
    return result


def depth_process(img):
    result = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][j] = (img[i][j][0] + img[i][j][1] * 256 + img[i][j][2] * 256 * 256) / (256 * 256 * 256 - 1)
    return result


def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


# 0.4 ~ 3m/s * 3.6km/h speed
class CarlaEnvironmentWrapper:
    def __init__(self, stack_frames=1, img_width=84, img_height=84, control_onput=1, rank=0, port=2000, throttle=0.35,
                 randomization=False, preprocess="origin", render=False):

        self.rank = rank
        self.randomization = randomization
        self.imagetype = preprocess

        self.no_progress_start = 50

        self.observation_space = gym.spaces.Box(-1, 1, shape=(stack_frames, img_height, img_width))
        self.action_space = gym.spaces.Box(-1, 1, shape=(control_onput,))

        self.stack_frames = stack_frames
        self.control_output = control_onput
        self.throttle = throttle

        # server configuration
        self.port = port
        self.host = "localhost"
        self.render = render
        self.map = "/Game/Maps/Town01"  # "/Game/Maps/Town02"

        # Camera
        self.height = img_height
        self.width = img_width

        # action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(control_onput,))

        logging.disable(40)

        self.episodes = 0

        self.last_steer = 0.0
        self._open_server()
        time.sleep(2)
        np.random.seed(int(time.time()))

        self.game = CarlaClient(self.host, self.port, timeout=99999999)
        self.set_settings()
        self.game.connect()
        # print("connected")

        scene = self.game.load_settings(self.settings)
        # positions = scene.player_start_spots
        # positions = 147
        self.num_pos = 147
        self.index = 0
        self.step_per_episode = 0

    def set_settings(self):
        self.settings = CarlaSettings()
        vehicle = 0
        pedestrain = 0
        weather = 1
        # if self.randomization:
        #     weather = np.random.choice([1, 2, 5, 7])
        # vehicle = np.random.choice(range(10*self.rank, 10*self.rank+10))
        # pedestrain = np.random.choice(range(10*self.rank, 10*self.rank+10))
        self.settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=vehicle,
            NumberOfPedestrians=pedestrain,
            WeatherId=weather,
            QualityLevel="Low")

        # add cameras
        # if self.imagetype == "origin":
        camera = Camera('Camera')
        # elif self.imagetype == "depth":
        cameraDepth = Camera('CameraDepth', PostProcessing='Depth')
        # else:
        cameraSeg = Camera('CameraSeg', PostProcessing='SemanticSegmentation')

        camera.set_image_size(self.width, self.height)
        x = 2.0
        y = 0.0
        z = 1.4
        fov = 90.0
        # print("x={0}, y={1}, z={2}, fov={3}".format(x, y, z, fov))

        camera.set(FOV=fov)
        camera.set_position(x=x, y=y, z=z)
        camera.set_rotation(0.0, 0.0, 0.0)

        cameraSeg.set_image_size(self.width, self.height)
        x = 2.0
        y = 0.0
        z = 1.4
        fov = 90.0
        # print("x={0}, y={1}, z={2}, fov={3}".format(x, y, z, fov))

        cameraSeg.set(FOV=fov)
        cameraSeg.set_position(x=x, y=y, z=z)
        cameraSeg.set_rotation(0.0, 0.0, 0.0)

        self.settings.add_sensor(cameraSeg)

    def reset(self, random_start=None):

        # if (self.episodes+1) % 10 == 0:
        #     self.game.disconnect()
        #     self.init_settings()

        # self.set_settings()
        # start_position = np.random.choice(range(self.num_pos))
        s = [74, 75, 113, 114]

        if random_start is not None:
            start_position = random_start
        else:
            if self.randomization == True:
                if self.rank <= 1:
                    start_position = np.random.choice(s[self.rank])
                else:
                    start_position = np.random.choice(range(147))
            else:
                if self.rank <= 1:
                    start_position = np.random.choice(s[self.rank])
                else:
                    start_position = np.random.choice(range(147))

        trials = 0
        self.step_per_episode = 0

        while trials < 10:
            try:
                self.game.start_episode(start_position)
                print("start with %d" % start_position)
                break
            except:
                self.end()
                time.sleep(5)
                self._open_server()
                self.game.connect()
                self.set_settings()
                scene = self.game.load_settings(self.settings)
                self.game.start_episode(start_position)

                print("start with %d" % start_position)
                trials += 1
                print("Connect Failed, try %d times" % trials)

        # if (self.episodes+1) % 10 == 0:
        #     self.game.load_settings(self.settings)

        self.episodes += 1

        self.state = None

        # Fixed throttle and brake
        self.control = VehicleControl()
        self.control.throttle = self.throttle
        self.control.brake = 0
        self.control.steer = 0
        self.control.hand_brake = False
        self.control.reverse = False

        for i in range(14):
            self.game.send_control(self.control)
            # measurements, sensor_data = self.game.read_data()
        self.measurements, sensor_data = self.game.read_data()

        return self._update_state(self.control)[0]

    def step(self, action):

        control = self.measurements.player_measurements.autopilot_control
        self.control.steer = control.steer
        self.game.send_control(self.control)
        # # discrete
        # steers = [0, -0.25, -0.12, -0.05, 0.05, 0.12, 0.25]
        # steer = steers[action]
        #
        # # continuous
        # # steer = action[0]
        # self.step_per_episode += 1
        #
        # self.control.steer = steer
        # if self.control_output >= 2:
        #     action[1] += 1
        #     action[1] /= 2
        #     self.control.throttle = action[1]
        #     self.control.brake = 0
        # if self.control_output >= 3:
        #     pass
        #     # if action[1] > 0:
        #     #     self.control.throttle = action[1]
        #     #     self.control.brake = 0
        #     # else:
        #     #     self.control.throttle = 0
        #     #     self.control.brake = -action[1]
        #
        # self.game.send_control(self.control)
        return self._update_state(self.control)

    def end(self):
        self.game.disconnect()
        self._close_server()

    def _open_server(self):
        # log_path = path.join(logger.experiments_path, "CARLA_LOG_{}.txt".format(self.port))
        self.port = get_open_port()
        with open("/dev/null", "wb") as out:
            cmd = ""
            if self.render == False:
                cmd = "SDL_HINT_CUDA_DEVICE=0 SDL_VIDEODRIVER=offscreen "
            cmd += path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh ') + self.map + \
                   " -benchmark" + " -carla-server" + " -fps=10" + " -world-port={}".format(self.port) + \
                   " -windowed -ResX={} -ResY={}".format(10, 10) + \
                   " -carla-no-hud"
            print(cmd)
            self.server = subprocess.Popen([cmd], shell=True, preexec_fn=os.setsid, stdout=out, stderr=out)

    def _close_server(self):
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)

    def _update_state(self, action):
        # get measurements and observations
        measurements = []
        while type(measurements) == list:
            measurements, sensor_data = self.game.read_data()
        self.measurements = measurements

        self.location = (measurements.player_measurements.transform.location.x,
                         measurements.player_measurements.transform.location.y,
                         measurements.player_measurements.transform.location.z)

        speed = measurements.player_measurements.forward_speed
        is_stuck = False
        # is_stuck = (self.no_progress_start < self.step_per_episode) and (speed < 0.1)
        done = measurements.player_measurements.collision_vehicles != 0 \
               or measurements.player_measurements.collision_pedestrians != 0 \
               or measurements.player_measurements.collision_other != 0 \
               or is_stuck \
               or measurements.player_measurements.intersection_offroad > 0.05
        # or measurements.player_measurements.intersection_otherlane > 0.70 \

        # if self.imagetype == "origin":
        img = image_converter.to_rgb_array(sensor_data.get('Camera', None))
        # elif self.imagetype == "depth":
        imgDepth = image_converter.depth_to_logarithmic_grayscale(sensor_data.get('CameraDepth', None))
        # else:
            # img = image_converter.labels_to_cityscapes_palette(sensor_data.get('Camera', None))
        imgSeg = image_converter.labels_to_array(sensor_data.get('CameraSeg', None))

        # import matplotlib.pyplot as plt
        # import pillow
        # plt.imshow(img)
        # plt.show()
        # plt.savefig("12.jpg")
        # cv2.imwrite("%d.jpg" % 11, img)
        # cv2.imshow("img", img)
        # cv2.waitKey(1)
        # print(img)

        if self.index < 100000:
            imgSeg = map_process(imgSeg)
            cv2.imwrite("data/%d-depth-%f.png" % (self.index, self.control.steer), imgDepth)
            cv2.imwrite("data/%d-%f.png" % (self.index, self.control.steer), img)
            cv2.imwrite("data/%d-seg-%f.png" % (self.index, self.control.steer), imgSeg)
            self.index = self.index + 1

        # if self.imagetype == "origin":
        #     if self.render == True:
        #         cv2.imshow("img", img)
        #         cv2.waitKey(1)
        #     img = img / 125 - 1
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
        # elif self.imagetype == "depth":
        #     img = depth_process(img)
        #     if self.render == True:
        #         cv2.imshow("tmp", img)
        #         cv2.waitKey(1)
        # else:
            # print(np.any(img==6))
            # if self.render == True:
            #     tmp = to_color(img) / 255
            #     cv2.imshow("tmp", tmp)
            #     cv2.waitKey(1)
            # img /= 10
            # print(img)

        # print(img)
        if done == True:
            reward = -1
        else:
            if self.control_output == 1:
                reward = 0.5 * (1 - np.abs(self.control.steer) * 1.5)
            else:
                reward = (speed / 100) * (1 - np.abs(self.control.steer) * 1)

        self.last_steer = action.steer

        self.state = img.reshape(3, self.height, self.width)
        # if self.state is None:
        #     self.state = np.stack((img, ) * self.stack_frames, axis=0)
        # else:
        #     img = img.reshape((1, self.height, self.width))
        #     self.state = np.append(self.state[1:, :, :], img, axis=0)

        return self.state, reward, done, None


if __name__ == "__main__":
    env = CarlaEnvironmentWrapper(1, 84, 84, port=45000, throttle=0.35, control_onput=1, randomization=True,
                                  preprocess="origin", render=True)
    img = env.reset()
    for i in range(100002):
        # print(img.shape)
        # cv2.imshow("img", (img[0, :, :]+1)/2)
        # img = (img[0, :, :] + 1) * 127.5
        # cv2.imshow("img", img)
        # cv2.waitKey(1)
        # print(img[0])
        # print(img[0].shape)
        # print(img[1])
        img, _, done, info = env.step(0)
        # print(_)
        if done == True:
            break
    env.end()
