from pyGameWorld import PGWorld, ToolPicker
# from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld.viewer import *
import json
import pygame as pg
import os
import random
import imageio
import numpy as np
import math

class ToolEnv:
    def __init__(self, environment = 0, json_dir = "environment/Trials/Original/", shaped = True):
        self.json_dir = json_dir
        self.tp = None
        self.worlds = os.listdir(self.json_dir)
        print(self.worlds)

        # relevant env:
        with open(self.json_dir + self.worlds[environment], 'r') as f:
            btr = json.load(f)
        self.tp = ToolPicker(btr)
        self.dims = (600, 600)
        self.action_dict = {0 : "obj1",
                            1 : "obj2",
                            2 : "obj3"}
        self.shaped = shaped


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

    def middle_of(self, pts_list):
        npts = len(pts_list)
        middle = [0, 0]
        for pt in pts_list:
            middle[0] += pt[0]
            middle[1] += pt[1]
        middle[0] /= npts
        middle[1] /= npts
        return middle

    def dist(self, pt_a, pt_b):
        dist = (pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2
        dist = math.sqrt(dist)
        return dist

    def step(self, action: np.array, display = False) -> float:
        tool_select = action[0]
        position = (action[1 : ] + 1) * (self.dims[0] / 2) #shift and scale from (-1, 1) to (0, 600)
        position = position.tolist()
        assert tool_select <= 2 and tool_select >= 0
        position = self.clip(position)

        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname="obj1", position=(90, 400), maxtime=20.)
        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname=self.action_dict[action[0]], position=action[1], maxtime=20.)
        path_dict, success, time_to_success, wd = self.tp.observeFullPlacementPath(toolname=self.action_dict[tool_select], position=position, maxtime=20., returnDict=True)
        if success is None:
            return 0.0
        if not success:
            if not self.shaped:
                return 0
            #shaped reward
            # demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)
            goal = wd["objects"]["Goal"]
            middle_of_goal = self.middle_of(goal["points"])
            baseline_distance = self.dist(path_dict["Ball"][0][0], middle_of_goal)
            min_distance = min([self.dist(pt, middle_of_goal) for pt in path_dict["Ball"][0]])
            reward = 1 - min_distance / baseline_distance
        else:
            reward = 1.0

        # print(reward)
        # print(tool_select, position, success)

        # TODO: get proper reward (by looking at where the ball is relative to the goal
        self.state = path_dict
        self.wd = wd
        #path_dict["Ball"] contains trajectory of the ball through time
        if display:
            demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)
        #perchance derive from path_dict the rewaard
        return reward

    def render(self, step):
        sc = drawPathSingleImageWithTools(self.tp, self.state, pathSize=3, lighten_amt=.5, worlddict = self.wd, with_tools=True)
        pg.image.save(sc, f"results/{step}.png")
        #TODO replace with actual byte array


# env = ToolEnv(json_dir = "./Trials/Original/")
# env.reset()
# action = np.array([0, 90, 400])
# print(env.step(action))
# env.render()
