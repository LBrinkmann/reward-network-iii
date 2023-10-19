import random
import gym
from gym import spaces
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
import time

#from environment import Reward_Network
from environment import Environment

class Reward_Network(gym.Env):

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

        self.MAX_STEP = params['n_steps']  # 8

        # network info
        self.id = self.network['network_id']
        self.nodes = [n['node_num'] for n in self.network['nodes']]
        self.action_space = self.network['edges']
        self.possible_rewards = list(set([e['reward'] for e in self.network['edges']]))
        # self.possible_rewards = params['rewards']#[-100, -20, 0, 20, 140]
        self.reward_range = (min(self.possible_rewards) * self.MAX_STEP, max(self.possible_rewards) * self.MAX_STEP)

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

class RuleAgent:
    """
    Rule Agent class
    """

    def __init__(self, networks, strategy, params):
        """
        Initializes a Rule Agent object, that follows a specified strategy

        Args:
            networks (list): list of Reward_Network objects
            strategy (str): solving strategy name
            params (dict): parameters to solve networks eg n_steps or possible rewards
        """

        # assert tests
        self.solutions = []
        assert strategy in ['myopic', 'take_first_loss', 'random'], \
            f'Strategy name must be one of {["myopic", "take_first_loss", "random"]}, got {strategy}'

        self.networks = networks
        self.strategy = strategy
        self.params = params
        self.n_large_losses = self.params['n_losses']
        self.min_reward = min(self.params['rewards'])

        # colors for plot
        self.colors = {'myopic': 'skyblue', 'take_first_loss': 'orangered',
                       'random': 'springgreen'}

    def select_action(self, possible_actions, possible_actions_rewards):
        """
        We are in a current state S. Given the possible actions from S and the rewards
        associated to them this method returns the action to select (based on the current
        solving strategy)

        Args:
            possible_actions (np.array): array containing next possible states (expressed with node numbers)
            possible_actions_rewards (np.array): array containing rewards of next possible states

        Returns:
            (np.array): selected action
        """

        if self.strategy == 'take_first_loss':
            print(self.strategy, self.loss_counter, possible_actions_rewards)

        if self.strategy == 'random':
            return random.choice(possible_actions)

        # take first loss -> select among possible actions the one that gives best reward BUT
        # make sure to take a first big loss (-100 but can also change)
        if self.strategy == 'take_first_loss' and \
                self.loss_counter < self.n_large_losses and \
                self.min_reward in possible_actions_rewards:

            self.loss_counter += 1

            if len(np.argwhere(possible_actions_rewards == self.min_reward)[
                       0]) != 2:  # that is, we have only one big loss in the possible actions
                return possible_actions[
                    np.argwhere(possible_actions_rewards == self.min_reward)[0][
                        0]]
            else:  # else if both actions lead to big loss pick a random one
                return possible_actions[random.choice(
                    np.argwhere(possible_actions_rewards == self.min_reward)[
                        0])]
        else:

            try:
                if not np.all(
                        possible_actions_rewards == possible_actions_rewards[
                            0]):
                    return possible_actions[np.argmax(possible_actions_rewards)]
                else:
                    return random.choice(possible_actions)
            except:
                print(f'Error in network {self.environment.id}')
                print(self.environment.action_space)

    def solve(self):
        """
        Ths method solves the given networks, with different constraints depending on the strategy.
        Returns solution in tabular form

        Args:
            network (Reward_Network object): a network with info on nodes,edges
        """
        self.solutions = []
        solution_columns = ["network_id", "strategy", "step",
                            "source_node", "current_node", "reward",
                            "total_reward"]
        for network in self.networks:

            if self.strategy == 'take_first_loss':
                self.loss_counter = 0  # to reset!

            # solution variables
            solution = []

            # network environment variables
            self.environment = Reward_Network(network, self.params)
            self.environment.reset()

            while self.environment.is_done == False:
                s = []
                obs = self.environment.observe()
                a = self.select_action(obs['actions_available'],
                                       obs['next_possible_rewards'])
                step = self.environment.step(a)
                s.append(self.environment.id)
                s.append(self.strategy)
                s.append(step['n_steps'])
                s.append(step['source_node'])
                s.append(step['current_node'])
                s.append(step['reward'])
                s.append(step['total_reward'])
                solution.append(s)

            solution_df = pd.DataFrame(solution, columns=solution_columns)
            self.solutions.append(solution_df)

        return pd.concat(self.solutions, ignore_index=True)

    def save_solutions_frontend(self):
        """
        This method saves the selected strategy solution of the networks to be used in the experiment frontend;
        solutions are saved in a JSON file with network id and associated list of moves
        """
        df = pd.concat(self.solutions, ignore_index=True)

        def add_source(x):
            a = x
            a.insert(0, 0)
            return a

        s = df.groupby(['network_id'])['current_node'].apply(
            list).reset_index(name='moves')
        s['moves'] = s['moves'].apply(add_source)
        obj = s.to_dict('records')

        return df, json.dumps(obj)


if __name__ == '__main__':
    gen_params = {}
    gen_params['n_networks'] = 500
    gen_params['n_losses'] = 3
    gen_params['rewards'] = [-50, 0, 100, 200, 400]
    gen_params['n_steps'] = 8

    with open('networks.json') as json_file:
        networks = json.load(json_file)

    with open("solutions_myopic.json", "r") as read_file:
        solutions = json.load(read_file)
    s = json.loads(solutions)
    with open('myopic.json', 'w') as f:
        json.dump(s, f)


    # R = RuleAgent(networks, 'random', gen_params)
    # R.solve()
    # df_solutions, solutions = R.save_solutions_frontend()
    # df_solutions.to_csv('df_solutions_random.csv',sep=',')
    # with open('solutions_random.json', 'w') as f:
    #     json.dump(solutions, f)

    # M = RuleAgent(networks, 'myopic', gen_params)
    # M.solve()
    # df_solutions, solutions = M.save_solutions_frontend()
    # df_solutions.to_csv('df_solutions_myopic.csv', sep=',')
    # with open('solutions_myopic.json', 'w') as f:
    #     json.dump(solutions, f)

    #L = RuleAgent(networks, 'myopic', gen_params)
    #L.solve()
    #df_solutions, solutions = L.save_solutions_frontend()
    #df_solutions.to_csv('df_solutions_loss.csv', sep=',')
    #with open('solutions_loss.json', 'w') as f:
    #    json.dump(solutions, f)