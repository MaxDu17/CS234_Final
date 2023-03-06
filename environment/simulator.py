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
        meaningful_objects = {k : v for k, v in btr["world"]["objects"].items() if v["density"] != 0}
        self.object_prior_dict = {}
        for key, value in meaningful_objects.items():
            if value["type"] == "Ball":
                x_lims = [value["position"][0] - value["radius"], value["position"][0] + value["radius"]]
                y_mean = value["position"][1]
            elif value["type"] == "Poly":
                x_lims = self.find_x_lims(value["vertices"])
                y_mean = self.find_y_mean(value["vertices"])
            elif value["type"] == "Compound":
                pt_list = list()
                for poly in value["polys"]:
                    pt_list.extend(poly)
                x_lims = self.find_x_lims(pt_list)
                y_mean = self.find_y_mean(pt_list)
            else:
                raise Exception("unregisetered object!")
            # recentering around [-1, 1]
            x_lims[0] = (x_lims[0] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            x_lims[1] = (x_lims[1] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            y_mean = (y_mean - (self.dims[1] / 2)) / (self.dims[1] / 2)
            self.object_prior_dict[key] = (x_lims, y_mean)

    def visualize_prior(self, sigma_x, sigma_y):
        img = np.array(self.tp.drawPathSingleImage())
        target_color = np.array([255, 0, 0])
        for object, (xlims, y_mean) in self.object_prior_dict.items():
            x_left = int(((xlims[0] - sigma_x) * (self.dims[0] / 2)) + (self.dims[0] / 2))
            x_right = int(((xlims[1] + sigma_x) * (self.dims[0] / 2)) + (self.dims[0] / 2))
            y_left = int(((y_mean - sigma_y) * (self.dims[1] / 2)) + (self.dims[1] / 2))
            y_right = int(((y_mean + sigma_y) * (self.dims[1] / 2)) + (self.dims[1] / 2))
            y_left = max(0, y_left)
            y_right = min(600, y_right)
            img[600 - y_right : 600 - y_left, x_left : x_right] = 0.5 * img[600 - y_right : 600 - y_left, x_left : x_right] + 0.5 * target_color
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

    def find_x_lims(self, pt_list):
        x_min = min([v[0] for v in pt_list])
        x_max = max([v[0] for v in pt_list])
        return [x_min, x_max]

    def find_y_mean(self, pt_list):
        return sum([v[1] for v in pt_list]) / len(pt_list)

    def reset(self):
        self.tp._reset_pyworld()
        self.last_path = None
        self.state = None


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
        position = np.clip(position, 0, self.dims[0])
        position = position.tolist()

        assert tool_select <= 2 and tool_select >= 0
        position = self.clip(position)

        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname="obj1", position=(90, 400), maxtime=20.)
        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname=self.action_dict[action[0]], position=action[1], maxtime=20.)
        path_dict, success, time_to_success, wd = self.tp.observeFullPlacementPath(toolname=self.action_dict[tool_select], position=position, maxtime=20., returnDict=True)
        if success is None:
            self.last_path = None
            self.state = None
            print("\t\t FAIL")
            return 0.0
        if not success:
            if not self.shaped:
                return 0.0
            #shaped reward
            # demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)
            goal = wd["objects"]["Goal"]
            middle_of_goal = self.middle_of(goal["points"])
            baseline_distance = self.dist(path_dict["Ball"][0][0], middle_of_goal)
            min_distance = min([self.dist(pt, middle_of_goal) for pt in path_dict["Ball"][0]])
            reward = 1 - min_distance / baseline_distance
        else:
            reward = 1.0

        self.last_path = path_dict
        self.state = wd
        assert self.last_path is not None and self.state is not None, "somehow you queried an invalid position"
        #path_dict["Ball"] contains trajectory of the ball through time
        if display:
            print('demoed')
            demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)

        #perchance derive from path_dict the rewaard
        return reward

    def render(self):
        if self.state is not None and self.last_path is not None:
            img_arr = self.tp._get_image_array(self.state, self.last_path, sample_ratio=5)
            return img_arr
        return None

#
# env = ToolEnv(json_dir = "./Trials/Original/", environment = 1)
# env.reset()
# env.visualize_prior(sigma_x = 0.1, sigma_y = 0.7)
# # action = np.array([0, 90, 400])
# action = np.array([0, 300, 500])
# while True:
#     print("yes")
#     env.step(action, display= True)
#     input("enter")

# print(env.step(action))
# env.render()
