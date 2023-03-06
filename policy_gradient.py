import numpy as np
import torch
import gym
import os
from network_utils import np2torch
from base_gaussian_policy import GaussianToolPolicy
from environment.simulator import ToolEnv
import matplotlib.pyplot as plt
import argparse
import imageio
import tqdm

class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, seed, exp_dir, name, batch_size, object_prior):
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
        self.batch_size = batch_size
        # self.lr = 3e-2
        self.lr = 0.5
        self.exp_dir = exp_dir
        self.name = name

    def init_policy(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = GaussianToolPolicy(ntools = 3, nsteps = args.epochs, object_prior = self.env.object_prior_dict, device = device) #arbitrary steps for now; should converge very quickly
        self.policy.to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)

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
            action = self.policy.act()
            if args.counterfactual:
                #COUNTERFACTUAL SAMPLING
                for i in range(3):
                    env.reset()
                    action[0] = i
                    reward = env.step(action)

                    episode_rewards.append(reward)

                    paths.append({
                        "reward": reward,
                        "action": action.copy(),
                    })
            else:
                env.reset()
                reward = env.step(action)

                episode_rewards.append(reward)

                paths.append({
                    "reward": reward,
                    "action": action.copy(),
                })

            t += 1

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
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # print(self.policy.log_std.grad) #should gbe all populated
        # print(self.policy.means.grad) #should be all populated
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
        success_eval = list()
        for t in range(args.epochs):
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
            self.policy.anneal_epsilon(t)

            # loggin
            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            print(msg)

            # RENDERING FOR US
            if t % args.eval_frq == 0:
                rwds, succ= self.evaluate(t)
                print("LOW NOISE EVAL: ", rwds, succ)
                success_eval.append(succ)

            # self.env.render(t)
        fig, ax = plt.subplots()
        print(success_eval)
        x_axis = np.arange(0, args.epochs, args.eval_frq)
        ax.plot(x_axis, success_eval)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Epochs")
        ax.set_title("Performance on Basic Environment")

        fig.savefig(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}.png")
        np.save(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}", success_eval)
        plt.show()

    def evaluate(self, step):
        #TODO: generate meaningful animations
        avg_reward = 0
        avg_success = 0
        writer = imageio.get_writer(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}_{step}.mp4", fps=20)

        for i in tqdm.tqdm(range(args.eval_trials)):
            self.env.reset()
            action = self.policy.act(low_noise = True)
            rwd = self.env.step(action, display = False)# (i % 5 == 0))
            avg_reward += rwd
            avg_success += 1 if rwd > 0.99 else 0

            img_stack = self.env.render()
            if img_stack is not None:
                for f in range(img_stack.shape[0]):
                    writer.append_data(img_stack[f])

        writer.close()

        return avg_reward / args.eval_trials, avg_success / args.eval_trials

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required = False,
        default=1,
        help="seed",
    )

    parser.add_argument(
        "--counterfactual",
        action='store_true',
        help="counterfactual",
    )

    parser.add_argument(
        "--shaped_reward",
        action='store_true',
        help="shaped reward",
    )

    parser.add_argument(
        "--object_prior",
        action='store_true',
        help="object prior",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100,
        help="training_length",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=10,
        help="batch size ",
    )

    parser.add_argument(
        "--eval_frq",
        type=int,
        required=False,
        default=5,
        help="batch size",
    )

    parser.add_argument(
        "--eval_trials",
        type=int,
        required=False,
        default=20,
        help="number of times to run during eval",
    )

    parser.add_argument(
        "--level",
        type=int,
        required=False,
        default=0,
        help="level of the game you play",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="full_algorithm",
        help="name of algorithm run",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        required=False,
        default="experiments/initial_tests",
        help="name of algorithm run",
    )

    args = parser.parse_args()
    env = ToolEnv(environment = args.level, shaped = args.shaped_reward)
    pg = PolicyGradient(env, seed = args.seed, exp_dir = args.exp_dir,
                        name = args.name, batch_size = args.batch_size,
                        object_prior = args.object_prior)
    pg.run()
