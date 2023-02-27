import numpy as np
import torch
import gym
import os
from network_utils import np2torch
from base_gaussian_policy import GaussianToolPolicy
from environment.simulator import ToolEnv

class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module

        You do not need to implement anything in this function. However,
        you will need to use self.discrete, self.observation_dim,
        self.action_dim, and self.lr in other methods.
        """
        # directory for training outputs

        # store hyperparameters
        self.seed = seed
        torch.manual_seed(self.seed)

        self.env = env
        # self.env.seed(self.seed)
        self.batch_size = 10
        self.lr = 3e-2


    def init_policy(self):
        self.policy = GaussianToolPolicy(ntools = 3, bounds = 600)
        self.policy.to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        """
        You don't have to change or use anything here.
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass


    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode_rewards = []
        paths = []
        t = 0
        while t < self.batch_size:
            env.reset()
            action = self.policy.act()
            reward = env.step(action)
            # print(action, reward)
            t += 1
            episode_rewards.append(reward)

            path = {
                "reward": reward,
                "action": action,
            }
            paths.append(path)
        return paths, episode_rewards


    def update_policy(self, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data.
        To compute the loss, you will need the log probabilities of the actions
        given the observations. Note that the policy's action_distribution
        method returns an instance of a subclass of
        torch.distributions.Distribution, and that object can be used to
        compute log probabilities.
        See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
        PyTorch optimizers will try to minimize the loss you compute, but you
        want to maximize the policy's performance.
        """
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        # actions = torch.tensor(actions)
        log_probs = self.policy.log_prob(actions) #should be batch size
        loss = -torch.dot(log_probs, advantages) / advantages.shape[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(loss)
        self.policy.print_repr()

        #######################################################
        #########          END YOUR CODE.          ############

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(100):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            actions = np.stack([path["action"] for path in paths])
            rewards = np.stack([path["reward"] for path in paths])
            returns = rewards #only true because it's one-step return

            # # advantage will depend on the baseline implementation
            # advantages = self.calculate_advantage(returns, observations)
            # # run training operations
            # if self.config.use_baseline:
            #     self.baseline_network.update_baseline(returns, observations)

            self.update_policy(actions, returns)

            # loggin
            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            print(msg)

            # RENDERING FOR US
            env.reset()
            action = self.policy.act()
            env.step(action, display = True)

            # self.env.render(t)

    def evaluate(self, env=None, num_episodes=1):
        pass

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        # model
        self.init_policy()
        self.init_averages()
        self.train()
        # record one game at the end

env = ToolEnv()
pg = PolicyGradient(env, seed = 1)
pg.run()
