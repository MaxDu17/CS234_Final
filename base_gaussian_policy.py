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
            self.eps_begin = 0.1
            self.sigma_x = 0.1 #how far to sample beyond x limits
            # TODO: change this back to 0.7
            self.sigma_y = 0.7 # 1.4 #how far to sample beyond the y mean
        else:
            self.eps_begin = 1

        ##########

        self.eps_end = 0.8
        self.epsilon = self.eps_begin
        self.nsteps = nsteps
        self.retry_state = None #are we in prior or normal model mode?
        self.prior_state = None #what object and where (up or down) are we sampling from?
        self.retry = False # are we in retry mode?

    def px_to_action(self, val):
        # helper function that scales pixels to the value
        recentered = val - 300
        return recentered / 300

    def anneal_epsilon(self, t):
        self.epsilon = self.eps_begin + (min(t, self.nsteps) / self.nsteps) * (self.eps_end - self.eps_begin)
        print("annealed!", self.epsilon)

    def reset_prior(self):
        # print('RESET PRIOR')
        self.retry_state = None
        self.prior_state = None
        self.retry = False
        # normal running: retry stays false, and everything else doesn't matter

    def hold(self):
        # print('HOLD HOLD HOLD')
        self.retry = True
        assert self.retry_state is not None
        assert self.retry_state != "prior" or self.prior_state is not None

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

        # We use the policy if we 1) are retrying the policy or 2) we are in epsilon and not retrying
        if (self.retry and self.retry_state == "policy") or (np.random.rand() < self.epsilon and not prior_only and not self.retry):
            if self.retry:
                print("\t\tRETRYING POLICY")
            assert self.retry_state is None or self.retry_state == "policy"
            place_dist = ptd.MultivariateNormal(sampled_dist_mean, torch.diag(torch.exp(sampled_dist_log_std)))
            sampled_placement = place_dist.sample()
            self.retry_state = "policy"
        else:
            assert (not self.retry) or self.retry_state == "prior" # we are either in normal mode, or forced prior
            assert (not self.retry) or self.prior_state is not None #if we are retrying prior, we need to keep track!
            if self.retry:
                print("\t\tRETRYING PRIOR", self.prior_state)
            selected_object_name = self.object_names[np.random.randint(0, len(self.object_prior))] if not self.retry else self.prior_state[0]
            selected_object = self.object_prior[selected_object_name] #pick the object to interact with

            x_value = np.random.rand() * (2 * self.sigma_x + selected_object[0][1] - selected_object[0][0]) + (selected_object[0][0] - self.sigma_x)
            if x_value < selected_object[0][0]:
                x_value = np.random.normal(selected_object[0][0], self.sigma_x)
            elif x_value > selected_object[0][1]:
                x_value = np.random.normal(selected_object[0][1], self.sigma_x)

            y_samp = np.random.normal(0, self.sigma_y)
            above = y_samp > 0 if not self.retry else self.prior_state[1]
            if above: #this logic is roundabout, but it helps us replay the prior selection
                y_value = selected_object[1][1] + abs(y_samp)
            else:
                y_value = selected_object[1][0] - abs(y_samp)

            sampled_placement = torch.tensor([x_value, y_value], device = self.device)
            # this won't update until we manually reset the retry_state flag
            # print([x_value, y_value])
            self.prior_state = (selected_object_name,above)
            self.retry_state = "prior" #this is what you did last

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
