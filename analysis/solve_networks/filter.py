import random
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import os
import glob
import logging
import json
import torch
import time

from environment import Environment


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


class Reward_Network:

    def __init__(self, network, params):
        """_summary_

        Args:
            network (dict): a single network object
            params (dict): parameters for solving the networks eg n_steps, possible rewards
        """

        # -------------
        # assert tests
        # -------------

        # reward network information from json file
        self.network = network

        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.N_NODES = 10
        self.MAX_STEP = params['n_steps']  # 8

        # network info
        self.id = network['network_id']
        self.starting_node = network['starting_node']
        self.nodes = [n['node_num'] for n in network['nodes']]
        self.action_space = network['edges']
        rewards = torch.tensor(params['rewards'])
        self.possible_rewards = {rewards[r].item(): r + 1 for r in range(len(rewards))}

        # self.possible_rewards = list(set([e['reward'] for e in self.network['edges']]))
        self.reward_range = (min(self.possible_rewards) * self.MAX_STEP, max(self.possible_rewards) * self.MAX_STEP)

        new_edges = restructure_edges(network)

        self.action_space_idx = torch.full((self.N_NODES, self.N_NODES), 0)
        source = torch.tensor(new_edges["source_num"])
        target = torch.tensor(new_edges["target_num"])
        reward = torch.tensor(new_edges["reward"])
        reward_idx = torch.tensor([self.possible_rewards[x.item()] for x in reward])
        self.action_space_idx[source, target] = reward_idx


    def retrieve_solution(self, moves: list):
        """
        Given a list of moves, trace back the solution with rewards for each step and final score
        :return:
        """

        #assert moves[0] == rn.starting_node, f'error, starting node not valid'

        res = dict((int(v), k) for k, v in self.possible_rewards.items())
        reward_history = []
        for i in range(self.MAX_STEP):
            if i == 0:
                reward_history.append(res[int(self.action_space_idx[rn.starting_node,moves[i+1]])])
            else:
                reward_history.append(res[int(self.action_space_idx[moves[i], moves[i + 1]])])

        return reward_history


    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward_balance = self.INIT_REWARD
        self.step_counter = self.INIT_STEP
        self.is_done = False

        # Set the current step to the starting node of the graph
        self.current_node = self.network['starting_node']  # self.G[0]['starting_node']
        logging.info(f'NETWORK {self.id} \n')
        logging.info(f'INIT: Reward balance {self.reward_balance}, n_steps done {self.step_counter}')

    def step(self, action):
        # Execute one time step within the environment
        # self._take_action(action)
        self.source_node = action['source_num']  # OR with _id alternatively
        self.reward_balance += action['reward']
        self.current_node = action['target_num']
        self.step_counter += 1

        if self.step_counter == self.MAX_STEP:  # 8:
            self.is_done = True

        return {'source_node': self.source_node,
                'current_node': self.current_node,
                'reward': action['reward'],
                'total_reward': self.reward_balance,
                'n_steps': self.step_counter,
                'done': self.is_done}

    def get_state(self):
        """
        this function returns the current state of the environment.
        State information given by this funciton is less detailed compared
        to the observation.
        """
        return {'current_node': self.current_node,
                'total_reward': self.reward_balance,
                'n_steps': self.step_counter,
                'done': self.is_done}

    def observe(self):
        """
        this function returns observation from the environment
        """
        return {'current_node': self.current_node,
                'actions_available': [n for n in self.action_space if n['source_num'] == self.current_node],
                'next_possible_nodes': np.asarray(
                    [n['target_num'] for n in self.action_space if n['source_num'] == self.current_node]),
                'next_possible_rewards': np.asarray(
                    [n['reward'] for n in self.action_space if n['source_num'] == self.current_node]),
                'total_reward': self.reward_balance,
                'n_steps': self.step_counter,
                'done': self.is_done}


if __name__ == '__main__':

    gen_params = {}
    gen_params['n_networks'] = 500
    gen_params['n_losses'] = 3
    gen_params['rewards'] = [-50, 0, 100, 200, 400]
    gen_params['n_steps'] = 8

    check1_counter = 0
    check2_counter = 0

    n_to_keep = []
    solutions_to_keep = []
    solutions_loss_to_keep = []

    with open('networks_25_05_2023.json') as json_file:
        networks = json.load(json_file)
    with open("solutions_myopic_25_05_2023.json", "r") as read_file:
        solutions = json.load(read_file)
    with open("solutions_loss_25_05_2023.json", "r") as read_file:
        solutions_loss = json.load(read_file)

    print(len(networks))
    for n in networks:
        rn = Reward_Network(n, gen_params)

        # 2) check myopic solutions, how many solutions have a large loss in them?
        # Note: change first node in solution moves ot be starting node
        solution_moves = [s for s in solutions if s['network_id'] == rn.id][0]["moves"]
        # print([rn.starting_node] + solution_moves[1:])
        # print(rn.retrieve_solution(solution_moves))
        if -50 in rn.retrieve_solution(solution_moves):
            check2_counter += 1

            # 1) check that when starting in level 0 there is the option to stay in level 0, in other
            # words no starting node with both edges going to level 1 (-50 points in both)
            if sum(rn.action_space_idx[rn.starting_node, :] == 1) == 2:
                check1_counter += 1

        else:
            # original placement
            # n_to_keep.append(n)
            # solutions_to_keep.append([s for s in solutions if s['network_id'] == rn.id][0])
            # solutions_loss_to_keep.append([s for s in solutions_loss if s['network_id'] == rn.id][0])

            # new! compare points from myopic and points from loss
            solution_myopic = [s for s in solutions if s['network_id'] == rn.id][0]["moves"]
            solution_loss = [s for s in solutions_loss if s['network_id'] == rn.id][0]["moves"]
            print('reward myopic: ',rn.retrieve_solution(solution_myopic), "- reward loss: ", rn.retrieve_solution(solution_loss))
            if (rn.retrieve_solution(solution_loss)[0] == 100 and rn.retrieve_solution(solution_loss)[1] == -50 and rn.retrieve_solution(solution_myopic)[0] == 100) or \
                (rn.retrieve_solution(solution_loss)[0] == -50 and rn.retrieve_solution(solution_myopic)[0] == 100):
                n_to_keep.append(n)
                solutions_to_keep.append([s for s in solutions if s['network_id'] == rn.id][0])
                solutions_loss_to_keep.append([s for s in solutions_loss if s['network_id'] == rn.id][0])

            # print(f'solution myopic: {solution_myopic} -> '
            #       f'{rn.retrieve_solution(solution_myopic)} for {sum(rn.retrieve_solution(solution_myopic))} points')
            # print(f'solution loss: {solution_loss} -> '
            #       f'{rn.retrieve_solution(solution_loss)} for {sum(rn.retrieve_solution(solution_loss))} points')

    print(
        f"There are {check2_counter}/{len(networks)} networks ({round(check2_counter / len(networks) * 100, 2)}%) "
        f"to be discarded because the myopic solution has a -50 in the list of moves")
    print(
        f"Of these {check1_counter}/{check2_counter} networks ({round(check1_counter / check2_counter * 100, 2)}%) "
        f"to be discarded because in starting node both edges are -50")

    print(len(n_to_keep))


    # with open('networks_filtered_pilot5.json', 'w') as fout:
    #     json.dump(n_to_keep, fout)
    # with open('solutions_myopic_filtered_pilot5.json', 'w') as fout:
    #     json.dump(solutions_to_keep, fout)
    # with open('solutions_loss_filtered_pilot5.json', 'w') as fout:
    #     json.dump(solutions_loss_to_keep, fout)



    # with open("solutions_myopic_25_05_2023.json", "r") as read_file:
    #     solutions = json.load(read_file)
    # s = json.loads(solutions)
    # with open('myopic.json', 'w') as f:
    #     json.dump(s, f)
