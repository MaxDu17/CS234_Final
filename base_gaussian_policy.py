import torch

import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np


class GaussianToolPolicy(nn.Module):
    def __init__(self, ntools, nsteps):
        nn.Module.__init__(self)
        self.ntools = ntools
        # assuming that bounds is a single number, and the environment is a square

        self.tool_distribution = nn.Parameter(torch.ones(self.ntools))
        self.log_std = nn.Parameter(-1.79 * torch.ones(self.ntools, 2)) #start wide

        self.prior = nn.Parameter(torch.tensor([-0.66, 0.33]) , requires_grad = False) # [100, 215] is the real ball
        self.prior_stdev = nn.Parameter(torch.tensor([-3.0, -2.0]), requires_grad = False)

        self.means = nn.Parameter(torch.zeros(self.ntools, 2)) # start in the middle

        self.eps_begin = 0.1
        self.eps_end = 1
        self.epsilon = self.eps_begin
        self.nsteps = nsteps

    def px_to_action(self, val):
        # helper function that scales pixels to the value
        recentered = val - 300
        return recentered / 300

    def anneal_epsilon(self, t):
        self.epsilon = self.eps_begin + (min(t, self.nsteps) / self.nsteps) * (self.eps_end - self.eps_begin)
        print("annealed!", self.epsilon)

    def act(self, obs = None): #does not take an observation
        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        sampled_tool = tool_dist.sample()
        sampled_dist_mean = self.means[sampled_tool]
        sampled_dist_log_std = self.log_std[sampled_tool]

        if np.random.rand() < self.epsilon:
            print("policy!")
            place_dist = ptd.MultivariateNormal(sampled_dist_mean, torch.diag(torch.exp(sampled_dist_log_std)))
        else:
            place_dist = ptd.MultivariateNormal(self.prior, torch.diag(torch.exp(self.prior_stdev))) #INCORRECT
        sampled_placement = place_dist.sample()
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
