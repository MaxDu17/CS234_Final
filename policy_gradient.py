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
from collections import deque

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
        self.lr = 0.1 #0.1 #1.5 #0.5
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


    def sample_path(self, env, num_episodes=None, prior = False):
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
        # print("COLLECTING BATCH")
        niters = self.batch_size if num_episodes is None else num_episodes
        for t in tqdm.tqdm(range(niters)):
            if prior:
                action = self.policy.act(prior_only=True)
            else:
                action = self.policy.act()
            self.policy.hold()  # this just means that we will not change (either prior or policy)
            if args.counterfactual:
                #COUNTERFACTUAL SAMPLING
                for i in range(3):
                    env.reset()
                    action[0] = i
                    reward = env.step(action)
                    count = 0
                    while reward is None: #this is done for illegal moves
                        if prior:
                            action = self.policy.act(prior_only = True)
                        else:
                            action = self.policy.act()
                        action[0] = i
                        reward = env.step(action)
                        count += 1
                        if count == 100: #on th 10th try, reward automatically is 0
                            print('UNSUCCESSFUL, GIVING REWARD OF ZERO ***')
                            reward = 0.0
                            break
                    episode_rewards.append(reward)
                    paths.append({
                        "reward": reward,
                        "action": action.copy(),
                    })
                self.policy.reset_prior()  # reset the prior state after every sampling
            else:
                env.reset()
                reward = env.step(action)
                count = 0
                self.policy.hold()
                while reward is None:  # this is done for illegal moves
                    action = self.policy.act()
                    reward = env.step(action)
                    count += 1
                    if count == 100:  # on th 10th try, reward automatically is 0
                        print('UNSUCCESSFUL, GIVING REWARD OF ZERO ***')
                        reward = 0.0
                        break
                self.policy.reset_prior()
                episode_rewards.append(reward)

                paths.append({
                    "reward": reward,
                    "action": action.copy(),
                })
        return paths, episode_rewards


    def update_policy(self, actions, advantages):
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        # actions = torch.tensor(actions)
        log_probs = self.policy.log_prob(actions) #should be batch size
        loss = -torch.dot(log_probs, advantages) / advantages.shape[0]
        print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        # print(self.policy.log_std.grad) #should gbe all populated
        # print(self.policy.means.grad) #should be all populated
        self.optimizer.step()
        # print(loss)
        # self.policy.print_repr()
        return loss.item()

    def train(self):
        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        success_eval = list()

        paths, total_rewards = self.sample_path(self.env, num_episodes=10, prior = True)
        actions = np.stack([path["action"] for path in paths])
        returns = np.stack([path["reward"] for path in paths]) #only one step
        rewards_buffer = deque(maxlen=50)
        rewards_buffer.extend(returns.tolist())

        self.env.visualize_actions(actions, "action_distr.png")

        rwds, succ = self.evaluate(0)
        print("LOW NOISE EVAL: ", rwds, succ)
        success_eval.append(succ)

        print('BURN-IN')
        last_loss = 1000
        baseline = sum(rewards_buffer) / len(rewards_buffer) if args.baseline else 0
        for t in range(100):
            loss = self.update_policy(actions, returns - baseline)
            if last_loss - loss < 0.001:
                break
            last_loss = loss

        for t in range(args.epochs + 1):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            actions = np.stack([path["action"] for path in paths])
            returns = np.stack([path["reward"] for path in paths]) #only true because it's one-step return
            rewards_buffer.extend(returns.tolist())
            baseline = sum(rewards_buffer) / len(rewards_buffer) if args.baseline else 0
            print("MODIFIED REWARDS" , len(rewards_buffer), returns - baseline)
            self.update_policy(actions, returns - baseline)
            self.policy.anneal_epsilon(t)
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            print(msg)

            # RENDERING FOR US
            if t % args.eval_frq == 0 and t > 0:
                rwds, succ= self.evaluate(t)
                print("LOW NOISE EVAL: ", rwds, succ)
                success_eval.append(succ)
                torch.save(self.policy.state_dict(),
                           self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}_{t}.pt")  # saves everything from the state dictionary
            # self.env.render(t)
        fig, ax = plt.subplots()
        print(success_eval)
        x_axis = np.arange(0, args.epochs + 1, args.eval_frq)
        ax.plot(x_axis, success_eval)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Epochs")
        ax.set_title("Performance on Basic Environment")

        fig.savefig(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}.png")
        np.save(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}", success_eval)
        # plt.show()

    def evaluate(self, step):
        avg_reward = 0
        avg_success = 0
        writer = imageio.get_writer(self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}_{step}.mp4", fps=20)

        std = np.exp(self.policy.log_std.detach().cpu().numpy())
        means = self.policy.means.detach().cpu().numpy()
        self.env.visualize_distributions(means, std, self.exp_dir + f"/{self.name}_level{args.level}_{self.seed}_{step}.png")

        for i in tqdm.tqdm(range(args.eval_trials)):
            self.env.reset()
            rwd = None
            count = 0
            while rwd is None and count < 10:
                action = self.policy.act(low_noise = True)
                rwd = self.env.step(action, display = False)# (i % 5 == 0))
                count += 1
                if count == 10:
                    print("gave up!")
                    rwd = 0.0
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
        "--baseline",
        action='store_true',
        help="subtract average reward",
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
