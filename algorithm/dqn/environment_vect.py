# This file specifices the Reward Network Environment class in OpenAI Gym style.
# A Reward Network object can store and step in multiple networks at a time.
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################
import torch
import torch.nn.functional as F
from config_type import Config


def restructure_edges(network):
    """
    This function restructures the edges from list of dicts
    to one dict, to improve construction of edges matrix and
    env vectorization

    Args:
        network (list): list of dicts, where each dict is a Reward Network with nodes and edges' info

    Returns:
        new_edges (dict): dict with list for source id, target id and reward
    """

    new_edges = {"source_num": [], "target_num": [], "reward": []}
    for e in network["edges"]:
        new_edges["source_num"].append(e["source_num"])
        new_edges["target_num"].append(e["target_num"])
        new_edges["reward"].append(e["reward"])
    return new_edges


def extract_level(network):
    """
    This function extracts the level for each node in a network

    Args:
        network (_type_): _description_

    Returns:
        _type_: _description_
    """
    level = {}
    for e in network["nodes"]:
        level[e["node_num"]] = e["level"] + 1
    return level


class Reward_Network:
    def __init__(self, network, config: Config, device):
        """
        Initializes a reward network object given network(s) info in JSON format

        Args:
            network (list of dict): list of network information, where each network in list is a dict
                                    with keys nodes-edges-starting_node-total_reward
            config (Config): configuration parameters
            device: torch device
        """
        # observation shape from model config (determines whether to return obs in default format
        # or concatenating nodes features all in one dimension)
        self.observation_type = config.observation_type

        # reward network information from json file (can be just one network or multiple networks)
        self.network = network

        # torch device
        self.device = device

        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = config.n_rounds
        self.REWARDS = config.rewards
        self.N_REWARD_IDX = len(self.REWARDS) + 1  # (5 valid rewards + one to indicate no reward possible)
        self.N_NODES = config.n_nodes
        self.N_NETWORKS = len(self.network)
        self.TRAIN_BATCH_SIZE = config.train_batch_size
        assert self.TRAIN_BATCH_SIZE <= len(network), f'Batch size must be smaller or same as total number of networks'

        self.observation_shape = (self.N_NODES * self.N_REWARD_IDX)

        self.network_size_dict = {False: self.N_NETWORKS , True: self.TRAIN_BATCH_SIZE}

        # define node numbers (from 0 to x)
        self.nodes = torch.stack([torch.arange(self.N_NODES)] * self.N_NETWORKS, dim=0).to(self.device)

        # define starting nodes
        self.starting_nodes = torch.tensor(
            list(map(lambda n: n["starting_node"], self.network)), dtype=torch.long
        ).to(self.device)


        rewards = torch.tensor(self.REWARDS).to(self.device)
        rewards_norm = rewards.clone()
        if not all(rewards_norm == 0):  # non all-zero vector
            # linear rescale to range [0, 1]
            rewards_norm -= rewards_norm.min()  # bring the lower range to 0
            rewards_norm = torch.div(rewards_norm, rewards_norm.max())  # bring the upper range to 1
            # linear rescale to range [-1, 1]
            rewards_norm = 2 * rewards_norm - 1

        # define possible rewards along with corresponding reward index
        rewards_cuda = torch.tensor(self.REWARDS).to(self.device)
        self.possible_rewards = {rewards_cuda[r].item(): r + 1 for r in range(len(rewards_cuda))}

        # initialize action space ("reward index adjacency matrix")
        # 0 here means that no edge is present, all other indices from 1 to 5 indicate a reward
        # (the higher the index number, the higher the reward)
        self.action_space_idx = torch.full(
            (self.N_NETWORKS, self.N_NODES, self.N_NODES), 1
        ).long().to(self.device)

        new_edges = list(map(restructure_edges, network))
        self.network_idx = torch.arange(self.N_NETWORKS, dtype=torch.long).to(self.device)

        # initialize level information for all networks (organized in a n_networks x n_nodes x n_nodes matrix)
        # 4 possible levels (of current node in edge) + 0 value to indicate no edge possible
        levels = list(map(extract_level, network))
        self.level_space = torch.full(
            (self.N_NETWORKS, self.N_NODES, self.N_NODES), 0
        ).long().to(self.device)

        # build the action space and the level space matrix
        for n in range(self.N_NETWORKS):
            buffer_action_space = torch.full((self.N_NODES, self.N_NODES), 0).long().to(self.device)
            source = torch.tensor(new_edges[n]["source_num"]).long().to(self.device)
            target = torch.tensor(new_edges[n]["target_num"]).long().to(self.device)
            reward = torch.tensor(new_edges[n]["reward"]).long().to(self.device)
            reward_idx = torch.tensor([self.possible_rewards[x.item()] for x in reward]).to(self.device)
            #reward.apply_(lambda val: self.possible_rewards.get(val, 0))
            #buffer_action_space[source, target] = reward
            buffer_action_space[source, target] = reward_idx
            self.action_space_idx[n, :, :] = buffer_action_space

            buffer_level = torch.full((self.N_NODES, self.N_NODES), 0).long().to(self.device)
            where_edges_present = self.action_space_idx[n, :, :] != 0
            for node in range(self.N_NODES):
                buffer_level[node, where_edges_present[node, :]] = levels[n][node]
            self.level_space[n, :, :] = buffer_level

        prova_values = torch.tensor(list(self.possible_rewards.values()),dtype=torch.long).to(self.device)
        prova_keys = torch.tensor(list(self.possible_rewards.keys()), dtype=torch.long).to(self.device)

        # define reward map
        self.reward_map = torch.zeros(max(prova_values) + 1, dtype=torch.long).to(self.device)
        self.reward_map[prova_values] = torch.tensor(prova_keys, dtype=torch.long)

        # define reward in range -1,1 map
        self.reward_norm_map = self.reward_map.clone()
        self.reward_norm_map = self.reward_norm_map.float()
        self.reward_norm_map[1:] = rewards_norm
        print("environment_vect reward_norm_map", self.reward_norm_map)

        # boolean adjacency matrix
        self.edge_is_present = torch.squeeze(
            torch.unsqueeze(self.action_space_idx != 0, dim=-1)
        ).to(self.device)

    def reset(self):
        """
        Resets variables that keep track of env interaction metrics e.g. reward,step counter, loss counter,..
        at the end of each episode
        """
        # Reset the state of the environment to an initial state
        self.reward_balance = torch.full((self.N_NETWORKS, 1), self.INIT_REWARD, dtype=torch.float).to(self.device)
        self.step_counter = self.INIT_STEP
        self.big_loss_counter = torch.zeros((self.N_NETWORKS, 1), dtype=torch.long).to(self.device)
        self.is_done = False
        self.current_node = self.starting_nodes.clone()
        self.idx = torch.randint(0, self.N_NETWORKS, (self.TRAIN_BATCH_SIZE,)).to(self.device)


    def step(self, action):
        """
        Take a step in all environments given an action for each env;
        here action is given in the form of node index for each env
        action_i \in [0,1,2,3,4,5,6,7,8,9]

        Args:
            action (th.tensor): tensor of size n_networks x 1

        Returns:
            rewards (th.tensor): tensor of size n_networks x 1 with the corresponding reward obtained
                                 in each env for a specific action a

            (for DQN, if not at last round)
            next_obs (dict of th.tensor): observation of env(s) following action a
        """

        action = action.to(self.device)

        self.source_node = self.current_node[self.idx]

        # extract reward indices for each env
        rewards_idx = torch.unsqueeze(
            self.action_space_idx[self.network_idx[self.idx], self.current_node[self.idx], action], dim=-1
        ).to(self.device)

        # new! extract level indices for each env
        levels = torch.unsqueeze(
            self.level_space[self.network_idx[self.idx], self.current_node[self.idx], action], dim=-1
        ).to(self.device)

        # add to big loss counter if 1 present in rewards_idx
        self.big_loss_counter[self.idx] = torch.add(
            self.big_loss_counter[self.idx], (rewards_idx == 1).int()
        )

        # obtain numerical reward value corresponding to reward indices
        # rewards = self.reward_map[rewards_idx] (not normalized)
        rewards = self.reward_norm_map[rewards_idx]  # (normalized)
        # add rewards to reward balance
        self.reward_balance[self.idx] = torch.add(self.reward_balance[self.idx], rewards)

        # update the current node for all envs
        self.current_node[self.idx] = action
        # update step counter
        self.step_counter += 1
        if self.step_counter == self.MAX_STEP:
            self.is_done = True

        if not self.is_done:
            next_obs = self.observe()
        else:
            next_obs = None

        return next_obs, rewards, levels, self.is_done

    def get_state(self):
        """
        Returns the current state of the environment.
        State information given by this funciton is less detailed compared
        to the observation.
        """
        return {
            "current_node": self.current_node,
            "total_reward": self.reward_balance,
            "n_steps": self.step_counter,
            "done": self.is_done,
        }

    def observe(self):
        """
        Returns observation from the environment. The observation is made of a boolean mask indicating which
        actions are valid in each env + a main observation matrix.
        For each node in each environment the main observation matrix contains contatenated one hot encoded info on:
        - reward index
        - step counter
        - loss counter (has an edge with associated reward of -100 been taken yet)
        - level (what is the level of the current/starting node of an edge)

        was max_step + 1 before, now only max step because we cre about step 0 1 2 3 4 5 6 7 (in total 8 steps)

        Returns:
            obs (dict of th.tensor): main observation matrix (key=obs) + boolean mask (key=mask)
        """


        self.observation_matrix = torch.zeros(
            (
                self.TRAIN_BATCH_SIZE,
                self.N_NODES,
                self.N_REWARD_IDX,
            ),
            dtype=torch.long,
        ).to(self.device)

        self.next_rewards_idx = torch.squeeze(
            torch.unsqueeze(
                self.action_space_idx[self.network_idx[self.idx], self.current_node[self.idx], :], dim=-1
            )
        ).to(self.device)

        self.observation_matrix = F.one_hot(
            self.next_rewards_idx, num_classes=self.N_REWARD_IDX
        )

        # mask of next nodes
        #-----------------------
        # the second observation matrix (boolean mask indicating valid actions)
        self.next_nodes = torch.squeeze(
            torch.unsqueeze(
                self.edge_is_present[self.network_idx[self.idx], self.current_node[self.idx], :], dim=-1
            )
        )
        return {"mask": self.next_nodes,
                "obs": self.observation_matrix.reshape(
                    [self.TRAIN_BATCH_SIZE, (self.N_NODES * self.N_REWARD_IDX)])}
