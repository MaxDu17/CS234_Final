import torch

import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np


class GaussianToolPolicy(nn.Module):
    def __init__(self, ntools, nsteps, object_prior, device):
        nn.Module.__init__(self)
        self.ntools = ntools
        # assuming that bounds is a single number, and the environment is a square

        self.tool_distribution = nn.Parameter(torch.zeros(self.ntools))
        self.log_std = nn.Parameter(-1.79 * torch.ones(self.ntools, 2)) #start wide
        # self.log_std = nn.Parameter(-3 * torch.ones(self.ntools, 2)) #start wide

        # self.prior = nn.Parameter(torch.tensor([-0.66, 0.33]) , requires_grad = False) # [100, 215] is the real ball
        # self.prior_stdev = nn.Parameter(torch.tensor([-3.0, -2.0]), requires_grad = False)

        self.means = nn.Parameter(torch.zeros(self.ntools, 2)) # start in the middle

        ## TO CHANGE ##
        self.object_prior = object_prior
        self.object_names = list(self.object_prior.keys())
        self.device = device

        if object_prior is not None:
            self.eps_begin = 0.7
            self.sigma_x = 0.1 #how far to sample beyond x limits
            self.sigma_y = 0.7 #how far to sample beyond the y mean
        else:
            self.eps_begin = 1

        ##########

        self.eps_end = 0.7 #TODO: temporary disabling annealing
        self.epsilon = self.eps_begin
        self.nsteps = nsteps

    def px_to_action(self, val):
        # helper function that scales pixels to the value
        recentered = val - 300
        return recentered / 300

    def anneal_epsilon(self, t):
        self.epsilon = self.eps_begin + (min(t, self.nsteps) / self.nsteps) * (self.eps_end - self.eps_begin)
        print("annealed!", self.epsilon)

    def act(self, obs = None, low_noise = False, prior_only = False): #does not take an observation
        if low_noise:
            tool = torch.argmax(self.tool_distribution)
            place_dist = ptd.MultivariateNormal(self.means[tool], torch.diag(torch.exp(self.log_std[tool] - 5)))
            place_sample = place_dist.sample()
            action = np.zeros((3))
            action[0] = tool.item()
            action[1:] = place_sample.detach().cpu().numpy()

            return action

        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        sampled_tool = tool_dist.sample()
        sampled_dist_mean = self.means[sampled_tool]
        sampled_dist_log_std = self.log_std[sampled_tool]

        if np.random.rand() < self.epsilon and not prior_only:
            # print("REAL")
            place_dist = ptd.MultivariateNormal(sampled_dist_mean, torch.diag(torch.exp(sampled_dist_log_std)))
            sampled_placement = place_dist.sample()

        else:
            selected_object_name = self.object_names[np.random.randint(0, len(self.object_prior))]
            selected_object = self.object_prior[selected_object_name] #pick the object to interact with
            # print(selected_object_name)

            x_value = np.random.rand() * (2 * self.sigma_x + selected_object[0][1] - selected_object[0][0]) + (selected_object[0][0] - self.sigma_x)
            if x_value < selected_object[0][0]:
                x_value = np.random.normal(selected_object[0][0], self.sigma_x)
            elif x_value > selected_object[0][1]:
                x_value = np.random.normal(selected_object[0][1], self.sigma_x)

            y_samp = np.random.normal(0, self.sigma_y)
            if y_samp < 0: #select what's above or below the object
                y_value = selected_object[1][0] + y_samp
            else:
                y_value = selected_object[1][1] + y_samp
            sampled_placement = torch.tensor([x_value, y_value], device = self.device)
            # place_dist = ptd.MultivariateNormal(self.prior, torch.diag(torch.exp(self.prior_stdev))) #INCORRECT
        action = np.zeros((3))
        action[0] = sampled_tool.item()
        action[1 : ] = sampled_placement.cpu().numpy()
        return action #(sampled_tool.item(), sampled_placement.cpu().numpy().tolist())

    def log_prob(self, action):
        # tool, placement = action
        tool = action[:, 0].type(torch.long)
        placement = action[:, 1:]
        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        tool_log_prob = tool_dist.log_prob(tool)
        covs = torch.diag_embed(torch.exp(self.log_std[tool]), offset=0, dim1=-2, dim2=-1)
        place_dist = ptd.MultivariateNormal(self.means[tool], covs)
        placement_log_prob = place_dist.log_prob(placement)
        return tool_log_prob + placement_log_prob

    def print_repr(self):
        print("________________________________")
        print(self.log_std.detach().cpu().numpy())
        print(self.means.detach().cpu().numpy())
        print(self.tool_distribution.detach().cpu().numpy())
        print("________________________________")
