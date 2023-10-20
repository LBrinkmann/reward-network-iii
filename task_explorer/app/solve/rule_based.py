import json
import random

import numpy as np
import pandas as pd

from .environment import Reward_Network


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

        return json.dumps(obj)
