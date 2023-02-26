import torch

import torch
import torch.nn as nn
import torch.distributions as ptd


class GaussianToolPolicy(nn.Module):
    def __init__(self, ntools, bounds):
        nn.Module.__init__(self)
        self.ntools = ntools
        self.bounds = bounds
        # assuming that bounds is a single number, and the environment is a square

        self.tool_distribution = nn.Parameter(torch.ones(self.ntools))
        self.log_std = nn.Parameter(5 * torch.ones(self.ntools, 2)) #start wide
        self.means = nn.Parameter((self.bounds / 2) * torch.ones(self.ntools, 2))

    def act(self, obs = None): #does not take an observation
        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        sampled_tool = tool_dist.sample()
        sampled_dist_mean = self.means[sampled_tool]
        sampled_dist_log_std = self.log_std[sampled_tool]

        place_dist = ptd.MultivariateNormal(sampled_dist_mean, torch.diag(torch.exp(sampled_dist_log_std)))
        sampled_placement = place_dist.sample()
        #TODO: should be clipping to fit within the bounds
        return (sampled_tool.item(), sampled_placement.cpu().numpy().tolist())

    def log_prob(self, action):
        tool, placement = action
        tool_dist = ptd.categorical.Categorical(logits=self.tool_distribution)
        tool_log_prob = tool_dist.log_prob(tool).detach().cpu().numpy()
        place_dist = ptd.MultivariateNormal(self.means[tool], torch.diag(torch.exp(self.log_std[tool])))
        placement_log_prob = place_dist.log_prob(placement).detach().cpu().numpy()
        return tool_log_prob + placement_log_prob
