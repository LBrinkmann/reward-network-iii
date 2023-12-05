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
        print(" | ".join(f"{k}: {v}" for k, v in data.items()))


class Agent:
    def __init__(
            self, obs_dim: int, config: dict, action_dim: tuple, save_dir: str, device
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_networks = config.n_networks
        self.n_nodes = config.n_nodes
        self.batch_size = config.batch_size
        self.n_steps = config.n_rounds

        # torch device
        self.device = device

        # specify DNNs used by the agent in training and learning Q(s,a) from experience
        # to predict the most optimal action - we implement this in the Learn section
        # two DNNs - policy net with Q_{online} and target net with Q_{target}- that
        # independently approximate the optimal action-value function.
        # 21 with all info, 15 with no level info and 14 with no level and loss counter info
        self.observation_shape = config.observation_shape
        self.observation_type = config.observation_type
        self.observation_final_size = {'default': {'full': 21, 'no_level': 15, 'no_level_loss_counter': 16},
                                       'concatenated': {'full': 21 * config.n_nodes,
                                                        'no_level': 15 * config.n_nodes,
                                                        'no_level_loss_counter': 16 * config.n_nodes}}
        # check model type from config
        self.model_type = config.model_type
        if config.model_type == 'DQN':

            if config.observation_shape == 'default':
                input_size = (
                    config.n_networks,
                    config.n_nodes,
                    self.observation_final_size[config.observation_shape][config.observation_type],
                )
                hidden_size = (
                    config.n_networks,
                    config.n_nodes,
                    config.nn_hidden_layer_size,
                )
                # one q value for each action
                output_size = (
                    config.n_networks,
                    config.n_nodes,
                    1,
                )

            elif config.observation_shape == 'concatenated':
                input_size = (
                    config.n_networks,
                    self.observation_final_size[config.observation_shape][config.observation_type],
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

            self.policy_net = DQN(input_size, output_size, hidden_size)
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = DQN(input_size, output_size, hidden_size)
            self.target_net = self.target_net.to(self.device)

        elif config.model_type == 'RNN':

            input_size = (
                config.n_networks,
                1,
                self.observation_final_size[config.observation_shape][config.observation_type],
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
            print(input_size[-1], hidden_size[-1], output_size[-1])
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
                                                      verbose=True)
        self.loss_fn = th.nn.SmoothL1Loss(reduction="none")

        # specify output directory
        self.save_dir = save_dir

        # specify parameters for rule based
        # self.loss_counter = th.zeros(config.n_networks).int()
        # self.n_losses = th.full((config.n_networks,), 3).int()

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

    def set_rule_based_objects(self, n):
        """

        """
        # specify parameters for rule based
        self.loss_counter = th.zeros(n).int()
        self.n_losses = th.full((n,), 3).int()

    def reset_loss_counter(self):
        """
        this method resets the loss counter at the end of each episode for the "take_loss" strategy
        """
        self.loss_counter = th.zeros(self.n_networks).int()

    def act_rule_based(self, obs, strategy: str):
        """
        Given a observation, choose an action (explore) according to a solving strategy with no
        DQN involved

        Args: obs (dict with values of th.tensor): observation from the env(s) comprising of one hot encoded
        reward+step counter+ big loss counter and a valid action mask

        Returns:
            action (th.tensor): node index representing next nodes to move to for all envs
            strategy (string): name of rule based strategy to use, one between ["myopic","take_loss","random"]
        """

        obs['mask'] = obs['mask'].to(self.device)

        n = obs['obs'].shape[0]
        print('act rule based n: ', n)

        # reshape the observation appropriately (self.n_networks <-> n)
        if self.observation_shape == 'concatenated':
            obs["obs"] = obs['obs'].reshape(
                [n, self.n_nodes, self.observation_final_size['default'][self.observation_type]]).to(
                self.device)
        else:
            obs["obs"] = obs["obs"].to(self.device)

        # get the reward indices for each of the 10 nodes within each environment
        # current_possible_reward_idx = th.zeros((self.n_networks, 10)).type(th.long)
        current_possible_reward_idx = th.zeros((n, 10)).type(th.long)
        # TODO: check that the :6 is there to not incldue level and loss counter information
        splitted = th.split(th.nonzero(obs["obs"][:, :, :6]), 10)
        for i in range(len(splitted)):
            current_possible_reward_idx[i, :] = splitted[i][:, 2]

        if strategy == "myopic":
            action = th.unsqueeze(th.argmax(current_possible_reward_idx, dim=1), dim=-1)

        elif strategy == "take_loss":
            action = th.unsqueeze(th.argmax(current_possible_reward_idx, dim=1), dim=-1)
            # print(f'loss counter shape: {self.loss_counter.shape}')
            # print(f'losses shape: {self.n_losses.shape}')
            # that is, if there are still envs where loss counter < 3
            if not th.equal(self.loss_counter, self.n_losses):

                loss_envs = (self.loss_counter != self.n_losses).nonzero()
                # print(f"environments where loss counter is still <2", loss_envs.shape)
                if loss_envs is not None:
                    envs_where_loss_present = th.unique(
                        (current_possible_reward_idx[loss_envs[:, 0], :] == 1).nonzero()[:, 0])
                    # print("environment with loss counter <2 where there is a loss", envs_where_loss_present.shape)
                    indices_selected_losses = th.multinomial(
                        (current_possible_reward_idx[loss_envs[:, 0], :] == 1)[envs_where_loss_present, :].float(), 1)
                    # print("indices of selected loss actions", indices_selected_losses.shape)
                    loss_actions = current_possible_reward_idx[loss_envs[:, 0], :].gather(1, indices_selected_losses)
                    # print("actual actions", loss_actions.shape)
                    action[envs_where_loss_present, 0] = indices_selected_losses[:, 0]

                    indices_loss_counter = loss_envs[:, 0][th.isin(loss_envs[:, 0], envs_where_loss_present)]
                    # indices_loss_counter2 = th.arange(self.n_networks)[
                    #    th.isin(th.arange(self.n_networks), indices_loss_counter)]
                    indices_loss_counter2 = th.arange(n)[th.isin(th.arange(n), indices_loss_counter)]
                    # print("indices of envs to make +1 on loss counter", indices_loss_counter2.shape)
                    self.loss_counter[indices_loss_counter2] += 1

        elif strategy == "random":
            action = th.multinomial(obs["mask"].type(th.float), 1)

        return action[:, 0]

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

        print(f"START act method obs {str.upper(self.observation_shape)} shape:", obs["obs"].shape)
        print(f"START act method mask {str.upper(self.observation_shape)} shape:", obs["mask"].shape)

        # EXPLORE (select random action from the action space)
        random_actions = th.multinomial(obs["mask"].type(th.float), 1)
        # print(f'random actions {th.squeeze(random_actions,dim=-1)}')

        # reset hidden state for GRU!
        if self.model_type == 'RNN' and first_call:
            self.policy_net.reset_hidden()

        # EXPLOIT (select greedy action)
        # return Q values for each action in the action space A | S=s
        # (the policy net has already been initialized with correct obs['obs'] shape)
        if self.model_type == 'DQN':
            action_q_values = self.policy_net(obs["obs"])
            if self.observation_shape == 'concatenated':
                # here we reshape to add final dimension (either self.n_networks or n)
                action_q_values = action_q_values.reshape([n, self.n_nodes, 1])
            print("action q values DQN shape:", action_q_values.shape)


        elif self.model_type == 'RNN':

            if self.observation_shape == 'default':
                print(f"obs {str.upper(self.observation_shape)} shape before rearrange: ", obs['obs'].shape)
                obs['obs'] = einops.rearrange(obs['obs'], '(i n) o f -> n i (o f)', i=1)
                print(f"obs {str.upper(self.observation_shape)} shape after rearrange: ", obs['obs'].shape)
                action_q_values = self.policy_net(obs["obs"]).reshape([n, self.n_nodes, 1])

            if self.observation_shape == 'concatenated':
                print(f"obs {str.upper(self.observation_shape)} shape before rearrange: ", obs['obs'].shape)
                obs['obs'] = einops.rearrange(obs['obs'], '(i n) a -> n i a', i=1)
                print(f"obs {str.upper(self.observation_shape)} shape after rearrange: ", obs['obs'].shape)
                action_q_values = self.policy_net(obs["obs"]).reshape([n, self.n_nodes, 1])
            print("ACT STEP Q VALUES RNN shape:", action_q_values.shape)

        # if self.observation_shape == 'concatenated':
        #    # here we reshape to add final dimension
        #    action_q_values = action_q_values.reshape([self.n_networks,self.n_nodes,1])

        # apply masking to obtain Q values for each VALID action (invalid actions set to very low Q value)
        action_q_values = self.apply_mask(action_q_values, obs["mask"])
        print(action_q_values.shape)
        # select action with highest Q value
        greedy_actions = th.argmax(action_q_values, dim=1).to(self.device)
        print(f'greedy actions {greedy_actions.shape}')

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
        print("td estimate state shape: ", state.shape)
        print("td estimate state mask shape: ", state_mask.shape)

        if self.model_type == 'DQN':

            td_est = self.policy_net(state)
            print('td est after model: ', td_est.shape)

            if self.observation_shape == 'concatenated':
                td_est = th.unsqueeze(td_est, -1)
            print("td est final shape", td_est.shape)

        if self.model_type == 'RNN':
            # reset hidden state
            self.policy_net.reset_hidden()

            # reshape the state to (batch,sequence,features)
            state = einops.rearrange(state, 'b r n o f -> (b n) r (o f)')
            print('td est state reshaped before model: ', state.shape)

            td_est = self.policy_net(state)
            print('td est after model: ', td_est.shape)

            td_est = einops.rearrange(td_est, '(b n) r o -> b r n o',
                                      b=self.batch_size,
                                      r=self.n_steps,
                                      n=n,  # self.n_networks,
                                      o=self.n_nodes)
            print("td est final reshape RNN", td_est.shape)

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
        # state has dimensions batch_size,n_steps,n_networks,n_nodes,
        # length of one hot encoded observation info
        print('\n')
        print(f"td target state shape -> {state.shape}")
        print(f"td target state mask shape -> {state_mask.shape}")
        print(f"td target reward shape -> {reward.shape}")

        next_max_Q2 = th.zeros(state.shape[:3], device=self.device)
        print(f"td target next_max_Q2 shape -> {next_max_Q2.shape}")

        if self.model_type == 'DQN':
            # target q has dimensions batch_size,n_steps,n_networks,n_nodes,1
            target_Q = self.target_net(state)
            if self.observation_shape == 'concatenated':
                target_Q = th.unsqueeze(target_Q, -1)

        # reset hidden state for GRU!
        if self.model_type == 'RNN':
            # reset hidden state
            self.target_net.reset_hidden()
            # change state dimensions
            state = einops.rearrange(state, 'b r n o f -> (b n) r (o f)')
            print('td target state reshaped before model: ', state.shape)

            target_Q = self.target_net(state)
            print('td target targetQ: ', target_Q.shape)
            target_Q = einops.rearrange(target_Q, '(b n) r o -> b r n o',
                                        b=self.batch_size,
                                        r=self.n_steps,
                                        n=n,  # self.n_networks,
                                        o=self.n_nodes)
            print("target_Q final reshape RNN", target_Q.shape)

        # # target q has dimensions batch_size,n_steps,n_networks,n_nodes,1
        # target_Q = self.target_net(state)
        # if self.observation_shape == 'concatenated':
        #     target_Q = th.unsqueeze(target_Q, -1)

        target_Q = self.apply_mask(target_Q, state_mask)
        print(f"target_Q masked shape -> {target_Q.shape}")
        # next_Q has dimensions batch_size,(n_steps -1),n_networks,n_nodes,1
        # (we skip the first observation and set the future value for the terminal state to 0)
        next_Q = target_Q[:, 1:]
        print(f"next_Q shape -> {next_Q.shape}")

        # next_max_Q has dimension batch,steps,networks
        next_max_Q = th.squeeze(next_Q).max(-1)[0].detach()
        print(f"next_max_Q shape -> {next_max_Q.shape}")

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

        print(f"LEARN method td_est shape {td_est.shape}")
        print(f"LEARN method td_tgt shape {td_est.shape}")

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
        print(f"observation shape: {obs['obs'].shape}")
        print(f"observation mask shape: {obs['mask'].shape}")


        if exp_mode:
            actions = th.full((self.n_networks, self.n_steps), 0)

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

            print('\n')

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
    env = Reward_Network(networks_train, config, DEVICE)

    env_test = Reward_Network(networks_test, config, DEVICE)


    # initialize Agent(s)
    AI_agent = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=config.save_dir,
        device=DEVICE,
    )
    
    # initialize Memory buffer
    Mem = Memory(
        device=DEVICE, size=config.memory_size, n_rounds=config.n_rounds
    )

    # initialize Logger(s) n_networks or train_batch_size from config
    logger = MetricLogger(
        "dqn", config.save_dir, config.train_batch_size, config.n_episodes, config.n_nodes, DEVICE
    )
    logger_test = MetricLogger(
        "dqn_test", config.save_dir, config.n_networks, config.n_episodes, config.n_nodes, DEVICE
    )

    metrics_df_list = []

    for e in range(config.n_episodes):
        print(f"----EPISODE {e + 1}---- \n")


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
        if (e + 1) % 100 == 0:
            print("<<<<TESTING!>>>>")
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
            print(f"MEMORY SAMPLE! Shape of memory sample obs -> {sample['obs'].shape}\n")
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
        print("\n")

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
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(**config)
    if WANDB_ENABLED:
        with wandb.init(project='reward-networks-iii', entity="chm-hci"):
            train_agent(config)
    else:
        train_agent(config)



