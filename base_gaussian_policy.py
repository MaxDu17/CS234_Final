import torch

import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np


class GaussianToolPolicy(nn.Module):
    def __init__(self, ntools, bounds):
        nn.Module.__init__(self)
        self.ntools = ntools
        self.bounds = bounds
        # assuming that bounds is a single number, and the environment is a square

        self.tool_distribution = nn.Parameter(torch.ones(self.ntools))
        self.log_std = nn.Parameter(-3 * torch.ones(self.ntools, 2)) #start wide

        prior = torch.ones(self.ntools, 2)
        #"cheating"
        prior[:, 0] *= -0.7 #(self.bounds / 4)
        prior[:, 1] *= 1.33 #(self.bounds / 2)
        self.means = nn.Parameter(prior)


    def act(self, obs = None): #does not take an observation
        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        sampled_tool = tool_dist.sample()
        sampled_dist_mean = self.means[sampled_tool]
        sampled_dist_log_std = self.log_std[sampled_tool]

        place_dist = ptd.MultivariateNormal(sampled_dist_mean, torch.diag(torch.exp(sampled_dist_log_std)))
        sampled_placement = place_dist.sample()
        #TODO: should be clipping to fit within the bounds
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
