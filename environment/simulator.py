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

import matplotlib.pyplot as plt
import matplotlib

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
        blocker_list = [self.find_x_lims(v["vertices"]) for v in btr["world"]["blocks"].values()]
        # TODO: make meaningful blocker exclusions
        import ipdb
        # ipdb.set_trace()

        self.img = np.array(self.tp.drawPathSingleImage(wd = None, path = None))

        self.object_prior_dict = {}
        for key, value in meaningful_objects.items():
            if value["type"] == "Ball":
                x_lims = [value["position"][0] - value["radius"], value["position"][0] + value["radius"]]
                y_lims = [value["position"][1] - value["radius"], value["position"][1] + value["radius"]]
            elif value["type"] == "Poly":
                x_lims = self.find_x_lims(value["vertices"])
                y_lims = self.find_y_lims(value["vertices"])
            elif value["type"] == "Compound":
                pt_list = list()
                for poly in value["polys"]:
                    pt_list.extend(poly)
                x_lims = self.find_x_lims(pt_list)
                y_lims = self.find_y_lims(pt_list)
            elif value["type"] == "Container":
                x_lims = self.find_x_lims(value["points"])
                y_lims = self.find_y_lims(value["points"])
            else:
                print(value["type"])
                raise Exception("unregisetered object!")
            # recentering around [-1, 1]
            x_lims[0] = self.norm(x_lims[0], self.dims[0])
            x_lims[1] = self.norm(x_lims[1], self.dims[0])
            y_lims[0] = self.norm(y_lims[0], self.dims[0])
            y_lims[1] = self.norm(y_lims[1], self.dims[0])
            # x_lims[0] = (x_lims[0] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            # x_lims[1] = (x_lims[1] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            # y_lims[0] = (y_lims[0] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            # y_lims[1] = (y_lims[1] - (self.dims[0] / 2)) / (self.dims[0] / 2)
            self.object_prior_dict[key] = (x_lims, y_lims)

    def norm(self, x, orig_scale):
        return ((x - orig_scale / 2) / (orig_scale / 2)) #centered between 0 and 1

    def denorm(self, x, orig_scale):
        return (x * (orig_scale / 2) + (orig_scale / 2))

    def visualize_prior(self, sigma_x, sigma_y):
        # img = np.array(self.tp.drawPathSingleImage())
        img = self.img.copy()
        target_color = np.array([255, 0, 0])
        for object, (xlims, ylims) in self.object_prior_dict.items():
            x_left = int(self.denorm(xlims[0] - sigma_x, self.dims[0]))
            x_right = int(self.denorm(xlims[1] + sigma_x, self.dims[0]))
            y_bottom = int(self.denorm(ylims[0] - sigma_y, self.dims[1]))
            y_top = int(self.denorm(ylims[1] + sigma_y, self.dims[1]))

            # x_left = int(((xlims[0] - sigma_x) * (self.dims[0] / 2)) + (self.dims[0] / 2))
            # x_right = int(((xlims[1] + sigma_x) * (self.dims[0] / 2)) + (self.dims[0] / 2))
            # y_bottom = int(((y_lims[0] - sigma_y) * (self.dims[1] / 2)) + (self.dims[1] / 2))
            # y_top = int(((y_lims[1] + sigma_y) * (self.dims[1] / 2)) + (self.dims[1] / 2))
            y_bottom = max(0, y_bottom)
            y_top = min(600, y_top)

            img[600 - y_top : 600 - int(self.denorm(ylims[1], self.dims[1])), x_left : x_right] = \
                0.5 * img[600 - y_top : 600 - int(self.denorm(ylims[1], self.dims[1])), x_left : x_right] + 0.5 * target_color
            img[600 - int(self.denorm(ylims[0], self.dims[1])) : 600 - y_bottom, x_left : x_right] = \
                0.5 * img[600 - int(self.denorm(ylims[0], self.dims[1])) : 600 - y_bottom, x_left : x_right] + 0.5 * target_color
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.savefig("prior.png")
        plt.show()

    def visualize_distributions(self, means, stdevs, save_dir = "policy.png"):
        # import ipdb
        # ipdb.set_trace()
        # img = np.array(self.tp.drawPathSingleImage())

        fig, ax = plt.subplots()
        ax.imshow(self.img)
        colors = ["cyan", "green", "purple"]
        for i in range(means.shape[0]):
            denormed_x = self.denorm(means[i][0], self.dims[0])
            denormed_y = self.denorm(means[i][1], self.dims[1])
            # ellipse = matplotlib.patches.Ellipse((means[i][0], means[i][1]), stdevs[i][0], stdevs[i][1], angle=0, alpha = 0.3)
            ellipse = matplotlib.patches.Ellipse((denormed_x, 600 - denormed_y),
                                                 stdevs[i][0] * self.dims[0],stdevs[i][1] * self.dims[1], angle=0, alpha = 0.3, color = colors[i])
            ax.add_patch(ellipse)
        plt.savefig(save_dir)
        plt.close()
        # plt.show()

    def find_x_lims(self, pt_list):
        x_min = int(min([v[0] for v in pt_list]))
        x_max = int(max([v[0] for v in pt_list]))
        return [x_min, x_max]

    def find_y_lims(self, pt_list):
        y_min = int(min([v[1] for v in pt_list]))
        y_max = int(max([v[1] for v in pt_list]))
        return [y_min, y_max]

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

    def step(self, action: np.array, display = False):
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
            return None
        if not success:
            if not self.shaped:
                return 0.0
            #shaped reward
            # demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)
            if "Goal" in wd["objects"]:
                goal = wd["objects"]["Goal"]
            else:
                goal_list = [v for k, v in wd["objects"].items() if v["type"] == "Goal"]
                assert len(goal_list) == 1
                goal = goal_list[0]
            try:
                middle_of_goal = self.middle_of(goal["points"])
            except KeyError:
                middle_of_goal = self.middle_of(goal["vertices"]) #janky, but this works

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

#TODO: object priors must be removed from forbidden regions

# env = ToolEnv(json_dir = "./Trials/Original/", environment = 4) #5 has blocker, and 3
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
