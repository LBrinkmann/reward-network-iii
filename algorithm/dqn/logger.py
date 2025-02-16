import time
import torch as th
from config_type import Config


class MetricLogger:
    def __init__(self, log_name, n_networks, config: Config, device):
        """
        Initialize logger object

        Args:
            log_name (str): name of the logger (train/test)
            n_networks (int): number of networks in the environment
            device: torch device
        """

        self.save_dir = config.save_dir
        self.log_name = log_name

        # params
        self.n_networks = n_networks
        self.n_episodes = config.n_episodes
        self.n_nodes = config.n_nodes
        self.n_steps = config.n_rounds
        self.device = device

        self.episode_metrics = {
                "reward_steps": [],
                "reward_episode": [],
                "reward_episode_all_envs": [],
                "level_steps": [],
                "level_episode": [],
                "level_episode_all_envs": []
            }

        if self.log_name == "train":
            self.episode_metrics["loss"] = []
            self.episode_metrics["q_learn"] = []
            # add step-wise q values tracker
            for i in range(self.n_steps):
                self.episode_metrics[f'q_mean_step_{i + 1}'] = []
                self.episode_metrics[f'q_max_step_{i + 1}'] = []

        # number of episodes to consider to calculate mean episode {current_metric}
        self.take_n_episodes = 5

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def init_episode(self):
        """
        Reset current metrics values
        """
        self.curr_ep_reward = 0.0
        self.reward_step_log = th.zeros(self.n_steps, self.n_networks).to(self.device)

        self.level_step_log = th.zeros(self.n_steps, self.n_networks).to(self.device)

        if self.log_name == "train":
            self.curr_ep_loss = 0.0
            self.curr_ep_q = 0.0
            self.curr_ep_loss_length = 0
            self.q_step_log = th.zeros(self.n_steps, self.n_networks, self.n_nodes).to(self.device)


    # def log_step(self, reward, reward2, q_step, step_number):
    def log_step(self, step_number: int, reward, level, q_step=None):
        """
        To be called at every transition within an episode, saves reward of the step
        and the aggregate functions of q values for each network-step

        Args:
            reward (th.tensor): reward obtained in current step in the env(s) (for all networks)
            level (th.tensor): levels transitioned to in current step in the env(s) (for all networks)
            q_step (th.tensor): q values of actions in current step in the env(s) (for all networks)
            step_number (int): current step in the env(s)
        """

        # change reward from -1,1 range to original values
        reward[reward == -1.0000] = -50
        mask = (reward > -0.8) & (reward < -0.7)
        reward[mask] = 0
        mask = (reward > -0.4) & (reward < -0.3)
        reward[mask] = 100
        mask = (reward > 0.1) & (reward < 0.2)
        reward[mask] = 200
        reward[reward == 1.0000] = 400

        # self.curr_ep_reward += reward
        self.reward_step_log[step_number, :] = reward[:, 0]
        # new! level, to be adjusted before (subtracting one so that level 0 is now the first level)
        ones = th.full((level.shape[0], 1), 1).to(self.device)
        level_adjusted = th.sub(level, ones, alpha=1)
        self.level_step_log[step_number, :] = level_adjusted[:, 0]

        if self.log_name == 'train' and q_step is not None:
            self.q_step_log[step_number, :, :] = q_step[:, :, 0].detach()

    def log_episode(self):
        """
        Store metrics'values at end of a single episode
        """

        # log the total reward obtained in the episode for each of the networks
        # self.episode_metrics['rewards'].append(self.curr_ep_reward)
        self.episode_metrics["reward_steps"].append(self.reward_step_log)
        self.episode_metrics["reward_episode"].append(
            th.squeeze(th.sum(self.reward_step_log, dim=0))
        )
        self.episode_metrics["reward_episode_all_envs"].append(
            th.mean(th.squeeze(th.sum(self.reward_step_log, dim=0))).item()
        )

        # new! log the level progression obtained in the episode for each of the networks
        self.episode_metrics["level_steps"].append(self.level_step_log)
        self.episode_metrics["level_episode"].append(
            th.squeeze(th.max(self.level_step_log, dim=0)[0])
        )
        self.episode_metrics["level_episode_all_envs"].append(
            th.mean(th.squeeze(th.max(self.level_step_log, dim=0)[0] )).item()
        )

        # log the loss value in the episode for each of the networks TODO: adapt to store when Learn method is called
        # self.episode_metrics['loss'].append(loss)

        if self.log_name == "train":
            # log the mean, min and max q value in the episode over all envs but FOR EACH STEP SEPARATELY
            # since we are mainly interested in the mean min and max of VALID actions we first get a mask to index
            # the valid actions in q_step_log

            mask_valid = self.q_step_log != th.finfo(self.q_step_log.dtype).min
            # now for each step calculate avg q value
            q_mean_steps = th.zeros(self.n_steps)
            q_min_steps = th.zeros(self.n_steps)
            q_max_steps = th.zeros(self.n_steps)
            for s in range(self.n_steps):
                q_mean_steps[s] = th.mean(self.q_step_log[s, :, :][mask_valid[s, :, :]])
                q_min_steps[s] = th.amin(self.q_step_log[s, :, :][mask_valid[s, :, :]])
                q_max_steps[s] = th.amax(self.q_step_log[s, :, :][mask_valid[s, :, :]])

            for s in range(self.n_steps):
                self.episode_metrics[f'q_mean_step_{s + 1}'].append(q_mean_steps[s].item())
                self.episode_metrics[f'q_max_step_{s + 1}'].append(q_max_steps[s].item())


        # reset values to zero
        self.init_episode()

    def log_episode_learn(self, q, loss):
        """
        Store metrics values at the call of Learn method

        Args:
            q (th.tensor): q values for each env
            loss (float): loss value
        """
        # log the q values from learn method
        self.episode_metrics["q_learn"].append(q)
        # log the loss value from learn method
        self.episode_metrics["loss"].append(loss)
