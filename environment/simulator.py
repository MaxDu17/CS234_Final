from pyGameWorld import PGWorld, ToolPicker
# from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld.viewer import *
import json
import pygame as pg
import os
import random
import imageio

class ToolEnv:
    def __init__(self, environment = 0):
        self.json_dir = "./Trials/Original/"
        self.tp = None
        self.worlds = os.listdir("./Trials/Original")
        print(self.worlds)

        # relevant env:
        with open(self.json_dir + self.worlds[environment], 'r') as f:
            btr = json.load(f)
        self.tp = ToolPicker(btr)
        self.dims = (600, 600)
        self.action_dict = {0 : "obj1",
                            1 : "obj2",
                            2 : "obj3"}


    def reset(self):
        self.tp._reset_pyworld()
        self.state = None
        self.wd = None

    def clip(self, arr):
        if arr[0] < 0:
            arr[0] = 0
        if arr[0] > self.dims[0]:
            arr[0] = self.dims[0]
        if arr[1] < 0:
            arr[1] = 0
        if arr[1] > self.dims[1]:
            arr[1] = self.dims[1]

        return arr

    # TODO: get proper reward
    def step(self, action):
        tool_select = action[0]
        position = action[1]
        assert tool_select < 2 and tool_select >= 0
        position = self.clip(position)

        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname="obj1", position=(90, 400), maxtime=20.)
        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname=self.action_dict[action[0]], position=action[1], maxtime=20.)
        path_dict, success, time_to_success, wd = self.tp.observeFullPlacementPath(toolname=self.action_dict[tool_select], position=position, maxtime=20., returnDict=True)
        self.state = path_dict
        self.wd = wd
        #path_dict["Ball"] contains trajectory of the ball through time
        demonstrateTPPlacement(self.tp, 'obj1', (90, 400))
        #perchance derive from path_dict the rewaard
        return success

    def render(self):
        sc = drawPathSingleImageWithTools(self.tp, self.state, pathSize=3, lighten_amt=.5, worlddict = self.wd, with_tools=True)
        pg.image.save(sc, "test.png")
        #TODO replace with actual byte array


env = ToolEnv()
env.reset()
action = (0, [90, 400])
print(env.step(action))
env.render()
