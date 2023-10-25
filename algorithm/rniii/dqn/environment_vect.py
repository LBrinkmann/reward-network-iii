# This file specifices the Reward Network Environment class in OpenAI Gym style.
# A Reward Network object can store and step in multiple networks at a time.
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################
import torch
import torch.nn.functional as F
import os
import json


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
    def __init__(self, network, observation_shape: str, observation_type: str, train_batch_size: int, device):
        """
        Initializes a reward network object given network(s) info in JSON format

        Args:
            network (list of dict): list of network information, where each network in list is a dict
                                    with keys nodes-edges-starting_node-total_reward
            observation_shape (str): specifies whether to return observation as (n_networks x n_nodes x n_features)
                                     or as concatenated (n_networks x (n_nodes x n_features)) matrix
            observation_type (str): specifies whether to return all features possible, or only part of the observation
                                    information
            train_batch_size (int): specifies how many of the total networks need to be sampled to become a training
                                    batch
            device: torch device
        """

        # -------------
        # assert tests TODO
        # -------------
        assert train_batch_size <= len(network), f'Batch size must be smaller or same as total number of networks'

        # observation shape from model config (determines whether to return obs in default format
        # or concatenating nodes features all in one dimension)
        self.observation_type = observation_type
        self.observation_shape = observation_shape

        # reward network information from json file (can be just one network or multiple networks)
        self.network = network

        # torch device
        self.device = device

        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = 8
        self.N_REWARD_IDX = 6  # (5 valid rewards + one to indicate no reward possible)
        self.N_NODES = 10
        # self.N_LEVELS = 5  # (4 valid levels + one to indicate no level possible)
        self.N_LEVELS = 6  # (5 valid levels + one to indicate no level possible)
        self.N_NETWORKS = len(self.network)
        self.TRAIN_BATCH_SIZE = train_batch_size

        self.network_size_dict = {False: self.N_NETWORKS , True: self.TRAIN_BATCH_SIZE}

        # define node numbers (from 0 to 9)
        self.nodes = torch.stack([torch.arange(10)] * self.N_NETWORKS, dim=0).to(self.device)

        # define starting nodes
        self.starting_nodes = torch.tensor(
            list(map(lambda n: n["starting_node"], self.network)), dtype=torch.long
        ).to(self.device)

        # new! normalize rewards to fall into range [-1,1]
        # see https://discuss.pytorch.org/t/normalize-vectors-to-1-1-or-0-1/6905/6
        rewards = torch.tensor([-50, 0, 100, 200, 400]).to(self.device)
        rewards_norm = rewards.clone()
        if not all(rewards_norm == 0):  # non all-zero vector
            # linear rescale to range [0, 1]
            rewards_norm -= rewards_norm.min()  # bring the lower range to 0
            rewards_norm = torch.div(rewards_norm, rewards_norm.max())  # bring the upper range to 1
            # linear rescale to range [-1, 1]
            rewards_norm = 2 * rewards_norm - 1

        # define possible rewards along with corresponding reward index
        rewards_cuda = torch.tensor([-50, 0, 100, 200, 400]).to(self.device)
        #self.possible_rewards = {rewards[r].item(): r + 1 for r in range(len(rewards))}
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
        #self.reward_map = torch.zeros(
        #    max(self.possible_rewards.values()) + 1, dtype=torch.long
        #).to(self.device)

        #self.reward_map[list(self.possible_rewards.values())] = torch.tensor(
        #    list(self.possible_rewards.keys()), dtype=torch.long
        #)
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
        self.step_counter = torch.full((self.N_NETWORKS, 1), self.INIT_STEP).to(self.device)
        self.big_loss_counter = torch.zeros((self.N_NETWORKS, 1), dtype=torch.long).to(self.device)
        self.is_done = False
        self.current_node = self.starting_nodes.clone()

        self.idx = torch.randint(0, self.N_NETWORKS, (self.TRAIN_BATCH_SIZE,)).to(self.device)

        # print("ENVIRONMENT INITIALIZED:")
        # print(f"- set of nodes of shape {self.nodes.shape}")
        # print(f"- action space of shape {self.action_space_idx.shape}")
        # print(f"- reward balance of shape {self.reward_balance.shape}")
        # print(f"- big loss counter of shape {self.big_loss_counter.shape}")
        # print(f"- step counter of shape {self.step_counter.shape}")
        # print(f"- current node of shape {self.current_node.shape}")
        # print("\n")

    def step(self, action, round_number: int, train_subset: bool = False):
        """
        Take a step in all environments given an action for each env;
        here action is given in the form of node index for each env
        action_i \in [0,1,2,3,4,5,6,7,8,9]

        Args:
            action (th.tensor): tensor of size n_networks x 1
            round_number (int): current round number at which the step is applied.
                            Relevant to decide if the next observation of envs after action
                            also needs to be returned or not
            train_subset (bool): flag to signal that we are taking a step for a subset of the networks, a subset
                                 os size train_batch_size

        Returns:
            rewards (th.tensor): tensor of size n_networks x 1 with the corresponding reward obtained
                                 in each env for a specific action a

            (for DQN, if not at last round)
            next_obs (dict of th.tensor): observation of env(s) following action a
        """

        action = action.to(self.device)

        if train_subset:
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
            #self.reward_balance[self.idx] = torch.add(self.reward_balance[self.idx], rewards)

            # update the current node for all envs
            self.current_node[self.idx] = action
            # update step counter
            self.step_counter[self.idx] = torch.add(self.step_counter[self.idx], 1)
            if torch.all(self.step_counter[self.idx] == 8):
                self.is_done = True


        else:
            self.source_node = self.current_node

            # extract reward indices for each env
            rewards_idx = torch.unsqueeze(
                self.action_space_idx[self.network_idx, self.current_node, action], dim=-1
            )

            # new! extract level indices for each env
            levels = torch.unsqueeze(
                self.level_space[self.network_idx, self.current_node, action], dim=-1
            ).to(self.device)

            # add to big loss counter if 1 present in rewards_idx
            self.big_loss_counter = torch.add(
                self.big_loss_counter, (rewards_idx == 1).int()
            )

            # obtain numerical reward value corresponding to reward indices
            # rewards = self.reward_map[rewards_idx] (not normalized)
            rewards = self.reward_norm_map[rewards_idx]  # (normalized)
            # add rewards to reward balance
            self.reward_balance = torch.add(self.reward_balance, rewards)

            # update the current node for all envs
            self.current_node = action
            # update step counter
            self.step_counter = torch.add(self.step_counter, 1)
            if torch.all(self.step_counter == 8):
                self.is_done = True

        # (relevant for DQN) if we are in the last step return only rewards,
        # else return also observation after action has been taken
        # if round_number != 7:
        #     next_obs = self.observe()
        #     return next_obs, rewards
        # else:
        #     return rewards

        if round_number != 7:
            next_obs = self.observe(train_subset)
            return next_obs, rewards, levels
        else:
            return rewards, levels

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

    def observe(self, train_subset: bool = False):
        """
        Returns observation from the environment. The observation is made of a boolean mask indicating which
        actions are valid in each env + a main observation matrix.
        For each node in each environment the main observation matrix contains contatenated one hot encoded info on:
        - reward index
        - step counter
        - loss counter (has an edge with associated reward of -100 been taken yet)
        - level (what is the level of the current/starting node of an edge)

        was max_step + 1 before, now only max step because we cre about step 0 1 2 3 4 5 6 7 (in total 8 steps)

        Args:
            train_subset (bool): whether to subsample set of networks during training or not
        Returns:
            obs (dict of th.tensor): main observation matrix (key=obs) + boolean mask (key=mask)
        """

        if train_subset:

            self.observation_matrix = torch.zeros(
                (
                    self.TRAIN_BATCH_SIZE,
                    self.N_NODES,
                    (self.N_REWARD_IDX + self.MAX_STEP + 1 + self.N_LEVELS),
                ),
                dtype=torch.long,
            ).to(self.device)

            self.next_rewards_idx = torch.squeeze(
                torch.unsqueeze(
                    self.action_space_idx[self.network_idx[self.idx], self.current_node[self.idx], :], dim=-1
                )
            ).to(self.device)

            self.next_edges_levels_idx = torch.squeeze(
                torch.unsqueeze(
                    self.level_space[self.network_idx[self.idx], self.current_node[self.idx], :], dim=-1
                )
            ).to(self.device)

        else:

            self.observation_matrix = torch.zeros(
                (
                    self.N_NETWORKS,
                    self.N_NODES,
                    (self.N_REWARD_IDX + self.MAX_STEP + 1 + self.N_LEVELS),
                ),
                dtype=torch.long,
            ).to(self.device)

            self.next_rewards_idx = torch.squeeze(
                torch.unsqueeze(
                    self.action_space_idx[self.network_idx, self.current_node, :], dim=-1
                )
            ).to(self.device)

            self.next_edges_levels_idx = torch.squeeze(
                torch.unsqueeze(
                    self.level_space[self.network_idx, self.current_node, :], dim=-1
                )
            ).to(self.device)

        # one hot encoding of reward_idx
        # print(f'one hot encoding of reward idx shape \n')
        # print(F.one_hot(self.next_rewards_idx,num_classes=self.N_REWARD_IDX).shape)
        # print(f'example one hot encoding of reward idx for node 3 of all networks \n')
        # print(self.next_rewards_idx )
        # print(F.one_hot(self.next_rewards_idx,num_classes=self.N_REWARD_IDX)[:,3,:])
        self.observation_matrix[:, :, : self.N_REWARD_IDX] = F.one_hot(
            self.next_rewards_idx, num_classes=self.N_REWARD_IDX
        )

        if train_subset:
            # one hot encoding of step counter
            self.observation_matrix[
            :, :, self.N_REWARD_IDX: (self.N_REWARD_IDX + self.MAX_STEP)
            ] = torch.repeat_interleave(
                F.one_hot(self.step_counter[self.idx], num_classes=self.MAX_STEP), self.N_NODES, dim=1
            )

            # big loss counter
            self.observation_matrix[
            :,
            :,
            (self.N_REWARD_IDX + self.MAX_STEP): (
                    self.N_REWARD_IDX + self.MAX_STEP + 1
            ),
            ] = torch.unsqueeze(
                torch.repeat_interleave(self.big_loss_counter[self.idx], self.N_NODES, dim=1), dim=-1
            )

        else:
            # one hot encoding of step counter
            self.observation_matrix[
            :, :, self.N_REWARD_IDX: (self.N_REWARD_IDX + self.MAX_STEP)
            ] = torch.repeat_interleave(
                F.one_hot(self.step_counter, num_classes=self.MAX_STEP), self.N_NODES, dim=1
            )

            # big loss counter
            self.observation_matrix[
            :,
            :,
            (self.N_REWARD_IDX + self.MAX_STEP): (
                    self.N_REWARD_IDX + self.MAX_STEP + 1
            ),
            ] = torch.unsqueeze(
                torch.repeat_interleave(self.big_loss_counter, self.N_NODES, dim=1), dim=-1
            )

        # one hot encoding of current levle node for an edge
        # self.observation_matrix[:, :, -5:] = F.one_hot(
        #    self.next_edges_levels_idx, num_classes=self.N_LEVELS
        # )
        self.observation_matrix[:, :, -6:] = F.one_hot(
            self.next_edges_levels_idx, num_classes=self.N_LEVELS
        )

        # mask of next nodes
        #-----------------------
        if train_subset:
            # the second observation matrix (boolean mask indicating valid actions)
            self.next_nodes = torch.squeeze(
                torch.unsqueeze(
                    self.edge_is_present[self.network_idx[self.idx], self.current_node[self.idx], :], dim=-1
                )
            )

        else:
            # the second observation matrix (boolean mask indicating valid actions)
            self.next_nodes = torch.squeeze(
                torch.unsqueeze(
                    self.edge_is_present[self.network_idx, self.current_node, :], dim=-1
                )
            )


        # OUTPUT
        if self.observation_shape == 'default':
            if self.observation_type == 'full':
                # normal observation
                return {"mask": self.next_nodes, "obs": self.observation_matrix}
            elif self.observation_type == 'no_level':
                # observation without level info
                return {"mask": self.next_nodes, "obs": self.observation_matrix[:, :, :-6]}
            elif self.observation_type == 'no_level_loss_counter':
                # observation without level and loss counter info
                return {"mask": self.next_nodes, "obs": self.observation_matrix[:, :, :-7]}
        else:
            if self.observation_type == 'full':
                # normal observation
                return {"mask": self.next_nodes,
                        "obs": self.observation_matrix.reshape(
                            [self.network_size_dict[train_subset],
                             (self.N_NODES * (self.N_REWARD_IDX + self.MAX_STEP + 1 + self.N_LEVELS))])}
                # "obs": self.observation_matrix.reshape([self.N_NETWORKS, (self.N_NODES*(self.N_REWARD_IDX + self.MAX_STEP + 1 + self.N_LEVELS))])}
            elif self.observation_type == 'no_level':
                # observation without level info
                return {"mask": self.next_nodes,
                        "obs": self.observation_matrix[:, :, :-6].reshape(
                            [self.network_size_dict[train_subset],
                             (self.N_NODES * (self.N_REWARD_IDX + self.MAX_STEP + 1))])}
                # "obs": self.observation_matrix[:, :, :-6].reshape( [self.N_NETWORKS, (self.N_NODES*(self.N_REWARD_IDX + self.MAX_STEP + 1))] )}
            elif self.observation_type == 'no_level_loss_counter':
                # observation without level and loss counter info
                return {"mask": self.next_nodes,
                        "obs": self.observation_matrix[:, :, :-7].reshape(
                            [self.network_size_dict[train_subset], (self.N_NODES * (self.N_REWARD_IDX + self.MAX_STEP))])}
                # "obs": self.observation_matrix[:, :, :-7].reshape( [self.N_NETWORKS, (self.N_NODES*(self.N_REWARD_IDX + self.MAX_STEP))] )}

    # For quick testing purposes, comment out if not needed
# with open('../../data/networks_test.json') as json_file:

# with open('../../data/networks_train.json') as json_file:
#     test = json.load(json_file)
# env_test = Reward_Network(test[10:15], observation_shape='default', observation_type='full', train_batch_size=5, device="cpu")
# print('starting nodes:', env_test.starting_nodes)
# print('rewards mapping (normalized):', env_test.reward_norm_map)
# print(env_test.action_space_idx[0,:,:])
# print(f"\n")

# env_test.reset()
# print("FIRST RESET")
# print("reward balance: ", env_test.reward_balance)
# print("step counter: ", env_test.step_counter)
# print("big loss counter: ", env_test.big_loss_counter)
# print("is done: ", env_test.is_done)
# print("current node: ", env_test.current_node)
# print("training random network idx: ", env_test.idx)
# print(f"\n")
#
# obs = env_test.observe(train_subset=True)
# print("FIRST OBSERVE")
# print("observation: ", obs["obs"].shape)
# print("mask: ", obs["mask"].shape)
# # print("observation: ", obs["obs"][0,:,:])
# # print("observation: ", obs["mask"][0,:])
# next_obs, rewards, levels = env_test.step(torch.tensor([2,8,2,8,2]), 0, train_subset=True)
# # print(next_obs['obs'].shape)
# print("rewards: ", rewards[:,0])
# print("levels: ", levels[:,0])

#observation_matrix_conc = obs["obs"].reshape([env_test.N_NETWORKS, env_test.N_NODES*21])
#print(observation_matrix_conc.shape)
#print(observation_matrix_conc[0,:-42])

#next_obs, rewards = env_test.step(torch.tensor([2,8,3]), 0)
#print(next_obs['obs'][:,:,:-6].shape)
#print(rewards)
#print(env_test.observe())
