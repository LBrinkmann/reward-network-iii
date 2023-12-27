# This file specifies the Deep Q Learning AI agent model to solve a Reward Network DAG
# See also: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# and: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
#
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################

import yaml
import json
import os

import argparse
import einops
import pandas as pd
import torch
import torch as th
import wandb

from environment_vect import Reward_Network
from memory import Memory
from nn import DQN, RNN
from logger import MetricLogger
from config_type import Config




# change string to compare os.environ with to enable ("enabled") or disable wandb
WANDB_ENABLED = os.environ.get("WANDB_MODE", "enabled") == "disabled"


def train():
    if WANDB_ENABLED:
        with wandb.init():
            config = Config(**wandb.config)
            train_agent(wandb.config)
    else:
        config = Config()
        train_agent(config)

def log(data, table=None, model=False):
    if WANDB_ENABLED:
        if table is not None:
            wandb.log({"metrics_table": table})
        else:
            wandb.log(data)
    else:
        if table is not None:
            pass
        else:
            print(" | ".join(f"{k}: {v}" for k, v in data.items()))


class Agent:
    def __init__(
            self, observation_shape: int, config: dict, action_dim: tuple, save_dir: str, device
    ):
        """
        Initializes an object of class Agent

        Args:
        obs_dim (int): number of elements present in the observation (2, action space observation + valid
        action mask)
        config (dict): a dict of all parameters and constants of the reward network problem (e.g. number
        of nodes, number of networks..)
        action_dim (tuple): shape of action space of one environment
        save_dir (str): path to folder where to save model checkpoints into
        device: torch device (cpu or cuda)
        """

        # specify environment parameters
        self.action_dim = action_dim
        self.observation_shape = observation_shape
        self.n_nodes = config.n_nodes
        self.batch_size = config.batch_size
        self.n_steps = config.n_rounds

        # torch device
        self.device = device

        # check model type from config
        self.model_type = config.model_type

        input_size = (
            config.n_networks,
            1,
            observation_shape,
        )
        hidden_size = (
            config.n_networks,
            config.n_nodes * config.nn_hidden_layer_size
        )
        # one q value for each action
        output_size = (
            config.n_networks,
            config.n_nodes
        )
        self.policy_net = RNN(input_size, output_size, hidden_size)
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = RNN(input_size, output_size, hidden_size)
        self.target_net = self.target_net.to(self.device)

        # specify \epsilon greedy policy exploration parameters (relevant in exploration)
        self.exploration_rate = 1
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # specify \gamma parameter (how far-sighted our agent is, 0.9 was default)
        self.gamma = 0.99

        # specify training loop parameters
        self.burnin = 10  # min. experiences before training
        self.learn_every = 5  # no. of experiences between updates to Q_online
        self.sync_every = (
            config.nn_update_frequency
        )  # 1e4  # no. of experiences between Q_target & Q_online sync
        self.save_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # specify which loss function and which optimizer to use (and their respective params)
        self.lr = config.learning_rate
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer,
                                                      step_size=config.lr_scheduler_step,
                                                      gamma=config.lr_scheduler_gamma,
                                                      verbose=False)
        self.loss_fn = th.nn.SmoothL1Loss(reduction="none")

        # specify output directory
        self.save_dir = save_dir

    @staticmethod
    def apply_mask(q_values, mask):
        """
        This method assigns a very low q value to the invalid actions in a network,
        as indicated by a mask provided with the env observation

        Args:
            q_values (th.tensor): estimated q values for all actions in the envs
            mask (th.tensor): boolean mask indicating with True the valid actions in all networks

        Returns:
            q_values (th.tensor): _description_
        """

        q_values[~mask] = th.finfo(q_values.dtype).min
        return q_values

    def act(self, obs, greedy_only=False, first_call=False, episode_number=None):
        """
        Given a observation, choose an epsilon-greedy action (explore) or use DNN to
        select the action which, given $S=s$, is associated to highest $Q(s,a)$

        Args: obs (dict with values of th.tensor): observation from the env(s) comprising one hot encoded
                                                   reward+step counter+ big loss counter and a valid action mask
              greedy_only (bool): a flag to indicate whether to use greedy actions only or not (relevant for
                                    test environments)
              first_call (bool): a flag to indicate whether the call to act method is the first call or not.
              episode_number (int): the current episode number

        Returns:
            action (th.tensor): node index representing next nodes to move to for all envs
            action_values (th.tensor): estimated q values for action
        """

        # assert tests
        # assert isinstance(obs, dict), f"Expected observation as dict"

        obs['mask'] = obs['mask'].to(self.device)
        obs["obs"] = obs["obs"].to(self.device)

        # new! n (can be 1000 - n_networks - or train_batch_size of 100)
        n = obs["obs"].shape[0]

        # EXPLORE (select random action from the action space)
        random_actions = th.multinomial(obs["mask"].type(th.float), 1)

        # reset hidden state for GRU!
        if first_call:
            self.policy_net.reset_hidden()

        obs['obs'] = einops.rearrange(obs['obs'], '(i n) a -> n i a', i=1)
        action_q_values = self.policy_net(obs["obs"]).reshape([n, self.n_nodes, 1])

        # apply masking to obtain Q values for each VALID action (invalid actions set to very low Q value)
        action_q_values = self.apply_mask(action_q_values, obs["mask"])
        # select action with highest Q value
        greedy_actions = th.argmax(action_q_values, dim=1).to(self.device)

        # select between random or greedy action in each env
        select_random = (
                th.rand(n, device=self.device)
                < self.exploration_rate
        ).long()


        if greedy_only:
            action = greedy_actions
        else:
            action = select_random * random_actions + (1 - select_random) * greedy_actions

        # fixed exploration rate OR
        # decrease exploration_rate not at each step (boolean flag to e.g. decay only every 1000th episodes)
        # self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        if (episode_number + 1) % 1000 == 0:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action[:, 0], action_q_values

    def td_estimate(self, state, state_mask, action):
        """
        This function returns the TD estimate for a (state,action) pair

        Args:
            state (dict of th.tensor): observation
            state_mask (th.tensor): boolean mask to the observation matrix
            action (th.tensor): actions taken in all envs from memory sample

        Returns:
            td_est: Q∗_online(s,a)
        """

        # we use the online model here we get Q_online(s,a)
        state = state.to(self.device)
        if len(state.shape) > 3:
            n = state.shape[2]

        # reset hidden state
        self.policy_net.reset_hidden()

        # reshape the state to (batch,sequence,features)
        state = einops.rearrange(state, 'b r n o f -> (b n) r (o f)')
        td_est = self.policy_net(state)
        td_est = einops.rearrange(td_est, '(b n) r o -> b r n o',
                                    b=self.batch_size,
                                    r=self.n_steps,
                                    n=n,  # self.n_networks,
                                    o=self.n_nodes)

        # apply masking (invalid actions set to very low Q value)
        td_est = self.apply_mask(td_est, state_mask)

        # select Q values for the respective actions from memory sample
        td_est_actions = (
            th.squeeze(td_est).gather(-1, th.unsqueeze(action, -1)).squeeze(-1)
        )
        return td_est_actions

    # note that we don't want to update target net parameters by backprop (hence the th.no_grad),
    # instead the online net parameters will take the place of the target net parameters periodically
    @th.no_grad()
    def td_target(self, reward, state, state_mask):
        """
        This method returns TD target - aggregation of current reward and the estimated Q∗ in the next state s'

        Args:
            reward (_type_): reward obtained at current observation
            state (_type_): observation corresponding to applying next action a'
            state_mask (_type_): boolean mask to the observation matrix

        Returns:
            td_tgt: estimated q values from target net
        """

        state = state.to(self.device)
        if len(state.shape) > 3:
            n = state.shape[2]

        next_max_Q2 = th.zeros(state.shape[:3], device=self.device)

        # reset hidden state
        self.target_net.reset_hidden()
        # change state dimensions
        state = einops.rearrange(state, 'b r n o f -> (b n) r (o f)')

        target_Q = self.target_net(state)
        target_Q = einops.rearrange(target_Q, '(b n) r o -> b r n o',
                                    b=self.batch_size,
                                    r=self.n_steps,
                                    n=n,  # self.n_networks,
                                    o=self.n_nodes)

        target_Q = self.apply_mask(target_Q, state_mask)
        # next_Q has dimensions batch_size,(n_steps -1),n_networks,n_nodes,1
        # (we skip the first observation and set the future value for the terminal state to 0)
        next_Q = target_Q[:, 1:]

        # next_max_Q has dimension batch,steps,networks
        next_max_Q = th.squeeze(next_Q).max(-1)[0].detach()

        next_max_Q2[:, :-1, :] = next_max_Q

        return th.squeeze(reward) + (self.gamma * next_max_Q2)

    def update_Q_online(self, td_estimate, td_target):
        """
        This function updates the parameters of the "online" DQN by means of backpropagation.
        The loss value is given by F.smooth_l1_loss(td_estimate - td_target)

        \theta_{online} <- \theta_{online} + \alpha((TD_estimate - TD_target))

        Args:
            td_estimate (_type_): q values as estimated from policy net
            td_target (_type_): q values as estimated from target net

        Returns:
            loss: loss value
        """

        # calculate loss, defined as SmoothL1Loss on (TD_estimate,TD_target),
        # then do gradient descent step to try to minimize loss
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()

        # we apply mean to get from dimension (batch_size,1) to 1 (scalar)
        loss.mean().backward()

        # truncate large gradients as in original DQN paper
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.mean().item()

    def sync_Q_target(self):
        """
        This function periodically copies \theta_online parameters
        to be the \theta_target parameters
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        """
        This function saves model checkpoints
        """
        save_path = os.path.join(
            self.save_dir,
            f"Reward_network_iii_model_{int(self.curr_step // self.save_every)}.chkpt",
        )
        th.save(
            dict(
                model=self.policy_net.state_dict(),
                exploration_rate=self.exploration_rate,
            ),
            save_path,
        )
        print(
            f"Reward_network_iii_model checkpoint saved to {save_path} at step {self.curr_step}"
        )

    def load_model(self, checkpoint_path: str):
        """
        This function loads a model checkpoint onto the policy net object
        """
        self.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])
        self.policy_net.eval()


    def learn(self, memory_sample):
        """
        Update online action value (Q) function with a batch of experiences.
        As we sample inputs from memory, we compute loss using TD estimate and TD target,
        then backpropagate this loss down Q_online to update its parameters θ_online

        Args:
            memory_sample (dict with values as th.tensors): sample from Memory buffer object, includes
            as keys 'obs','mask','action','reward'

        Returns:
            (th.tensor,float): estimated Q values + loss value
        """

        # if applicable update target net parameters
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # if applicable save model checkpoints
        # if self.curr_step % self.save_every == 0:
        #    self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Get TD Estimate (mask already applied in function)
        td_est = self.td_estimate(memory_sample["obs"], memory_sample["mask"], memory_sample["action"])
        # Get TD Target
        td_tgt = self.td_target(memory_sample["reward"], memory_sample["obs"], memory_sample["mask"])

        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def solve_loop(self, episode: int, n_rounds: int, train_mode: bool, exp_mode: bool, env, logger, mem=None):
        """
        This function solves all networks in a loop over n_rounds

        Args:
            e: (int) current episode number
            n_rounds: (int) the number of steps to solve networks in
            env: (Environment object)
            train_mode: bool flag to signal if we are training or testing
            exp_mode: bool flag to signal if we are solvign networks for the experiment or not
            logger: (Logger object) for metrics
            obs: (attribute of env object) observation matrix, including mask of valid actions
            mem: (Memory object)

        Returns:
            Saves metrics in logger object
        """

        # reset env(s)
        env.reset()
        # obtain first observation of the env(s)
        # obs = env.observe()
        obs = env.observe()


        if exp_mode:
            actions = th.full((obs.shape[0], self.n_steps), 0)

        for round_num in range(n_rounds):
            action, step_q_values = self.act(obs, greedy_only=train_mode, first_call=round_num == 0, episode_number=episode)

            next_obs, reward, level, is_done = env.step(action)

            # remember transitions in memory if a mem object is passed during function call
            # (that is, if we are in dqn)
            if mem is not None:
                mem.store(round_num, reward=reward, action=action, **obs)

            if not is_done:
                obs = next_obs


            logger.log_step(round_num, reward, level, step_q_values)

            if exp_mode:
                actions[:, round_num] = action

            if is_done:
                break

        if exp_mode:
            return actions

#######################################
# TRAINING FUNCTION(S)
#######################################
def train_agent(config=None):
    """
    Train AI agent to solve reward networks (using wandb)

    Args:
        config (dict): dict containing parameter values, data paths and
                       flag to run or not run hyperparameter tuning
    """

    # ---------Loading of the networks---------------------

    print(f"Loading train networks from file: {config.train_data_name}")
    # Load networks (train)
    with open(config.train_data_name) as json_file:
        networks_train = json.load(json_file)
    print(f"Number of networks loaded: {len(networks_train)}")
    # Load networks (test)
    with open(config.test_data_name) as json_file:
        networks_test = json.load(json_file)
    print(f"Number of networks loaded: {len(networks_test)}")

    # ---------Specify device (cpu or cuda)----------------
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ---------Start analysis------------------------------
    # initialize environment(s)
    env = Reward_Network(networks_train, network_batch=config.network_batch, config=config, device=DEVICE)

    env_test = Reward_Network(networks_test, network_batch=None, config=config, device=DEVICE)


    # initialize Agent(s)
    AI_agent = Agent(
        observation_shape=env.observation_shape,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=config.save_dir,
        device=DEVICE,
    )

    # initialize Memory buffer
    Mem = Memory(
        device=DEVICE, size=config.memory_size, n_rounds=config.n_rounds
    )

    # initialize Logger(s) n_networks or network_batch from config
    logger = MetricLogger(
        'train', config.network_batch, config, DEVICE
    )
    logger_test = MetricLogger(
        'test', config.n_networks, config, DEVICE
    )

    metrics_df_list = []

    for e in range(config.n_episodes):
        # train networks
        AI_agent.solve_loop(
            episode=e,
            n_rounds=config.n_rounds,
            train_mode=True,
            exp_mode=False,
            env=env,
            logger=logger,
            mem=Mem,
        )
        # new! learning rate scheduler
        AI_agent.scheduler.step()

        # --END OF EPISODE--
        Mem.finish_episode()
        logger.log_episode()

        # prepare logging info that all model types share
        metrics_log = {"episode": e + 1,
                       "avg_reward_all_envs": logger.episode_metrics['reward_episode_all_envs'][-1],
                       }
        for s in range(config.n_rounds):
            metrics_log[f'q_mean_step_{s + 1}'] = logger.episode_metrics[f'q_mean_step_{s + 1}'][-1]
            metrics_log[f'q_max_step_{s + 1}'] = logger.episode_metrics[f'q_max_step_{s + 1}'][-1]

        # test networks (every 100 episodes)
        if (e + 1) % config.test_period == 0:
            print(f"----EPISODE {e + 1}---- \n")
            AI_agent.solve_loop(
                episode=e,
                n_rounds=config.n_rounds,
                train_mode=False,
                exp_mode=False,
                env=env_test,
                logger=logger_test,
            )
            logger_test.log_episode()


            # add test rewards to wandb metrics
            metrics_log["test_avg_reward_all_envs"] = logger_test.episode_metrics['reward_episode_all_envs'][-1]
            # add test levels to wandb metrics
            metrics_log["test_avg_level_all_envs"] = logger_test.episode_metrics['level_episode_all_envs'][-1]
        else:
            metrics_log["test_avg_reward_all_envs"] = float("nan")
            metrics_log["test_avg_level_all_envs"] = float("nan")

        # take memory sample!
        sample = Mem.sample(config.batch_size, device=DEVICE)
        if sample is not None:
            # Learning step
            q, loss = AI_agent.learn(sample)

            # Send the current training result back to Wandb (if wandb enabled), else print metrics
            # (send only every 100 episodes)
            if (e + 1) % 100 == 0:
                # add batch loss to metrics to log
                metrics_log["batch_loss"] = loss
                log(metrics_log)

        else:
            if (e + 1) % 100 == 0:
                metrics_log["batch_loss"] = float("nan")
                log(metrics_log)

            print(f"Skip episode {e + 1}")

        metrics_df_list.append(metrics_log)

    # SAVE MODEL
    AI_agent.save()

    # Dataframe of all metrics
    metrics_df = pd.DataFrame(metrics_df_list)
    metrics_table = wandb.Table(dataframe=metrics_df)
    log([], table=metrics_table)


if __name__ == "__main__":

    # Load config parameter from yaml file specified in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="algorithm/params/dqn/single_run_v2.yml", help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(**config)
    if WANDB_ENABLED:
        with wandb.init(project='reward-networks-iii', entity="chm-hci"):
            train_agent(config)
    else:
        train_agent(config)



