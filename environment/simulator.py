from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
import json
import pygame as pg
import os
import random


class ToolEnv:
    def __init__(self):
        self.json_dir = "./Trials/Original/"
        self.tp = None
        self.worlds = os.listdir("./Trials/Original")
        print(self.worlds)

        # relevant env:
        with open(self.json_dir + self.worlds[0], 'r') as f:
            btr = json.load(f)
        self.tp = ToolPicker(btr)


    def reset(self):
        self.tp._reset_pyworld()
        #TODO: you're selecting from one environment and resetting.

    # TODO: get proper reward
    def step(self, action):
        path_dict, success, time_to_success = self.tp.observePlacementPath(toolname="obj1", position=(90, 400), maxtime=20.)
        demonstrateTPPlacement(self.tp, 'obj1', (90, 400))
        #perchance derive from path_dict the rewaard
        return success

    def get_obs(self):
        pass
        # def _get_image_array(self, worlddict, path, sample_ratio=1):
        #     if path is None:
        #         imgs = makeImageArrayNoPath(worlddict, self.maxTime / self.bts / sample_ratio)
        #     else:
        #         imgs = makeImageArray(worlddict, path, sample_ratio)
        #     imgdata = np.array([pg.surfarray.array3d(img).swapaxes(0, 1) for img in imgs])
        #     return imgdata

env = ToolEnv()
env.reset()
env.step(None)
