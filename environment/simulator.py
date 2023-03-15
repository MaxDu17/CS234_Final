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
from PIL import Image

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
        self.tool_x_lim_dict = {k : max([abs(x) for x in self.find_x_lims(v[0])]) for k, v in btr["tools"].items()} #find the largest radius
        self.tool_y_lim_dict = {k : max([abs(x) for x in self.find_y_lims(v[0])]) for k, v in btr["tools"].items()} #find the largest radius

        meaningful_objects = {k : v for k, v in btr["world"]["objects"].items() if v["density"] != 0}
        blocker_x_list = [self.find_x_lims(v["vertices"]) for v in btr["world"]["blocks"].values()]
        blocker_y_list = [self.find_y_lims(v["vertices"]) for v in btr["world"]["blocks"].values()]

        self.img = np.array(self.tp.drawPathSingleImage(wd = None, path = None))

        # BASELINE DISTANCE
        # this is a really big hack, but it works
        path_dict, success, time_to_success, wd = self.tp._ctx.call('getGWPathAndRotPlacement', self.tp._worlddict,
                                                                   self.tp._tools["obj1"], [-100, -100], 20,
                                                                   self.tp.bts, {}, {})

        if "Goal" in wd["objects"]:
            goal = wd["objects"]["Goal"]
        else:
            goal_list = [v for k, v in wd["objects"].items() if v["type"] == "Goal"]
            assert len(goal_list) == 1 #only supports one goal right now
            goal = goal_list[0]

        try:
            self.middle_of_goal = self.middle_of(goal["points"])
        except KeyError:
            self.middle_of_goal = self.middle_of(goal["vertices"])  # janky, but this works
        self.balls = [v for k, v in path_dict.items() if "Ball" in k]

        min_distances = list()
        for ball in self.balls:
            min_distances.append(min([self.dist(pt, self.middle_of_goal) for pt in ball[0]]))
        self.baseline_ball = min(min_distances)

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
            if self.inside_forbidden(x_lims, y_lims, blocker_x_list, blocker_y_list):
                print(x_lims, y_lims, blocker_x_list, blocker_y_list, "FORBIDDEN")
                continue
            x_lims[0] = self.norm(x_lims[0], self.dims[0])
            x_lims[1] = self.norm(x_lims[1], self.dims[0])
            y_lims[0] = self.norm(y_lims[0], self.dims[0])
            y_lims[1] = self.norm(y_lims[1], self.dims[0])

            self.object_prior_dict[key] = (x_lims, y_lims)

    def inside_forbidden(self, x_lims, y_lims, blocker_x_lims, blocker_y_lims):

        for bx_lim, by_lim in zip(blocker_x_lims, blocker_y_lims):
            if x_lims[0] > bx_lim[0] and x_lims[1] < bx_lim[1]:
                if y_lims[0] > by_lim[0] and y_lims[1] < by_lim[1] and by_lim[0] == 0 and by_lim[1] == 600:
                    return True
        return False

    def norm(self, x, orig_scale):
        return ((x - orig_scale / 2) / (orig_scale / 2)) #centered between 0 and 1

    def denorm(self, x, orig_scale):
        return (x * (orig_scale / 2) + (orig_scale / 2))

    def visualize_prior(self, sigma_x, sigma_y, savedir):
        # img = np.array(self.tp.drawPathSingleImage())
        img = self.img.copy()
        target_color = np.array([255, 255, 0])
        for object, (xlims, ylims) in self.object_prior_dict.items():
            x_left = int(self.denorm(xlims[0] - sigma_x, self.dims[0]))
            x_right = int(self.denorm(xlims[1] + sigma_x, self.dims[0]))
            y_bottom = int(self.denorm(ylims[0] - sigma_y, self.dims[1]))
            y_top = int(self.denorm(ylims[1] + sigma_y, self.dims[1]))
            print(object, y_bottom, y_top)
            y_bottom = max(0, y_bottom)
            y_top = min(600, y_top)

            img[600 - y_top : 600 - int(self.denorm(ylims[1], self.dims[1])), x_left : x_right] = \
                0.5 * img[600 - y_top : 600 - int(self.denorm(ylims[1], self.dims[1])), x_left : x_right] + 0.5 * target_color
            img[600 - int(self.denorm(ylims[0], self.dims[1])) : 600 - y_bottom, x_left : x_right] = \
                0.5 * img[600 - int(self.denorm(ylims[0], self.dims[1])) : 600 - y_bottom, x_left : x_right] + 0.5 * target_color
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(img)
        plt.savefig(savedir,bbox_inches='tight')
        # plt.show()

    def visualize_distributions(self, means, stdevs, save_dir = "policy.png"):
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

    def visualize_actions(self, actions, save_dir):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        colors = ["cyan", "green", "purple"]

        points_list = [(list(), list()), (list(), list()), (list(), list())]
        for action in actions:
            points_list[int(action[0])][0].append(self.denorm(action[1], self.dims[0]))
            points_list[int(action[0])][1].append(600 - self.denorm(action[2], self.dims[1])) # for visual purposes

        for i in range(len(points_list)):
            ax.scatter(points_list[i][0], points_list[i][1], color=colors[i], s=10)

        plt.savefig(save_dir)
        plt.close()

    def find_x_lims(self, pt_list):
        x_min = int(min([v[0] for v in pt_list]))
        x_max = int(max([v[0] for v in pt_list]))
        return [x_min, x_max]

    def find_y_lims(self, pt_list):
        y_min = int(min([v[1] for v in pt_list]))
        y_max = int(max([v[1] for v in pt_list]))
        return [y_min, y_max]

    def reset(self):
        self.tp._reset_pyworld()
        self.last_path = None
        self.state = None

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
        tool_name = self.action_dict[tool_select]
        position = (action[1 : ] + 1) * (self.dims[0] / 2) #shift and scale from (-1, 1) to (0, 600)
        # clips intelligently; does not allow tool to be illegally over the edge
        position = np.clip(position, (self.tool_x_lim_dict[tool_name], self.tool_y_lim_dict[tool_name]),
                           (self.dims[0] - self.tool_x_lim_dict[tool_name], self.dims[0] - self.tool_y_lim_dict[tool_name]))
        position = position.tolist()

        if position[1] > 600: #shouldn't go here
            import ipdb
            ipdb.set_trace()

        assert tool_select <= 2 and tool_select >= 0
        # position = self.clip(position)

        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname="obj1", position=(90, 400), maxtime=20.)
        # path_dict, success, time_to_success = self.tp.observePlacementPath(toolname=self.action_dict[action[0]], position=action[1], maxtime=20.)
        path_dict, success, time_to_success, wd = self.tp.observeFullPlacementPath(toolname=tool_name, position=position, maxtime=20., returnDict=True)
        if success is None:
            # print("FAILURE")
            return None
        if not success:
            if not self.shaped:
                return 0.0
            #shaped reward
            # demonstrateTPPlacement(self.tp, self.action_dict[tool_select], position)
            # if "Goal" in wd["objects"]:
            #     goal = wd["objects"]["Goal"]
            # else:
            #     goal_list = [v for k, v in wd["objects"].items() if v["type"] == "Goal"]
            #     assert len(goal_list) == 1
            #     goal = goal_list[0]
            # try:
            #     middle_of_goal = self.middle_of(goal["points"])
            # except KeyError:
            #     middle_of_goal = self.middle_of(goal["vertices"]) #janky, but this works
            # balls = [v for k, v in path_dict.items() if "Ball" in k]

            # baseline_distances = list()
            min_distances = list()
            for ball in self.balls:
                # baseline_distances.append(self.dist(ball[0][0], self.middle_of_goal))
                min_distances.append(min([self.dist(pt, self.middle_of_goal) for pt in ball[0]]))
            # if min(min_distances) > self.baseline_ball:
            #     import ipdb
            #     ipdb.set_trace()
            reward = 1 - min(min_distances) / self.baseline_ball
            # reward = 1 - min(min_distances) / min(baseline_distances)
        else:
            reward = 1.0

        self.last_path = path_dict
        self.state = wd
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

    def display_env_with_tools(self, savedir):
        data = pg.image.tostring(drawWorldWithTools(self.tp, backgroundOnly=False), "RGBA")
        img = Image.frombytes('RGBA', (750, 600), data)
        img.save(savedir)


if __name__ == "__main__":
    for i in range(18):
        print(i)
        env = ToolEnv(json_dir="./Trials/Original/", environment=i)  # 5 has blocker, and 3
        env.reset()
        if i == 12 or i == 13:
            sigma = 1.3
        else:
            sigma = 0.7
        env.visualize_prior(sigma_x = 0.1, sigma_y = sigma, savedir = f"../visuals/priors/prior_{i}.png")
        env.display_env_with_tools(savedir = f"../visuals/env/{i}.png")

# # x = env.tp._ctx.call('runGWPlacement', env.tp._worlddict, env.tp._tools["obj1"],
# #                               [-100, -100], 20, env.tp.bts, {}, {})
#
# path_dict, success, time_to_success, wd = env.tp._ctx.call('getGWPathAndRotPlacement', env.tp._worlddict,
#                               env.tp._tools["obj1"], [-100, -100], 20,
#                               env.tp.bts, {}, {})

# # action = np.array([0, 90, 400])
# action = np.array([0, 300, 500])
# while True:
#     print("yes")
#     env.step(action, display= True)
#     input("enter")

# print(env.step(action))
# env.render()
