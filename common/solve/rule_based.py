import os
import json
import random
import yaml
import argparse

import numpy as np
import pandas as pd
from pydantic import BaseModel

from common.solve.environment_solve import Reward_Network


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


class RuleAgentParams(BaseModel):
    n_steps: int
    n_losses: int
    rewards: list[int]
    solution_columns: list[str]


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
        assert strategy in [
            "myopic",
            "take_loss",
            "random",
        ], f'Strategy name must be one of {["myopic", "take_loss", "random"]}, got {strategy}'

        self.networks = networks
        self.strategy = strategy
        self.params = dict(RuleAgentParams(**params))
        self.min_reward = min(self.params["rewards"])

        # colors for plot
        self.colors = {
            "myopic": "skyblue",
            "take_loss": "orangered",
            "random": "springgreen",
        }

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

        # if self.strategy == "take_loss":
        #     print(self.strategy, self.loss_counter, possible_actions_rewards)

        if self.strategy == "random":
            return random.choice(possible_actions)

        # take first loss -> select among possible actions the one that gives best reward BUT
        # make sure to take a first big loss (-100 but can also change)
        if (
            self.strategy == "take_loss"
            and self.loss_counter < self.params["n_losses"]
            and self.min_reward in possible_actions_rewards
        ):

            self.loss_counter += 1

            if (
                len(np.argwhere(possible_actions_rewards == self.min_reward)[0]) != 2
            ):  # that is, we have only one big loss in the possible actions
                return possible_actions[
                    np.argwhere(possible_actions_rewards == self.min_reward)[0][0]
                ]
            else:  # else if both actions lead to big loss pick a random one
                return possible_actions[
                    random.choice(
                        np.argwhere(possible_actions_rewards == self.min_reward)[0]
                    )
                ]
        else:

            try:
                if not np.all(possible_actions_rewards == possible_actions_rewards[0]):
                    return possible_actions[np.argmax(possible_actions_rewards)]
                else:
                    return random.choice(possible_actions)
            except:
                print(f"Error in network {self.environment.id}")
                print(self.environment.action_space)

    def solve(self):
        """
        Ths method solves the given networks, with different constraints depending on the strategy.
        Returns solution in tabular form
        Args:
            network (Reward_Network object): a network with info on nodes,edges
        """
        solutions = []

        for network in self.networks:

            if self.strategy == "take_loss":
                self.loss_counter = 0  # to reset!

            # network environment variables
            self.environment = Reward_Network(network, self.params)
            self.environment.reset()

            step_counter = 0
            while not self.environment.is_done:
                obs = self.environment.observe()
                a = self.select_action(
                    obs["actions_available"], obs["next_possible_rewards"]
                )
                step = self.environment.step(a)
                solutions.append(
                    {
                        "network_id": self.environment.id,
                        "strategy": self.strategy,
                        "n_steps": step["n_steps"],
                        "source_node": step["source_node"],
                        "current_node": step["current_node"],
                        "level": step["level"],
                        "max_level": step["max_level"],
                        "reward": step["reward"],
                        "total_reward": step["total_reward"],
                        "step": step_counter,
                    }
                )
                step_counter += 1
        self.solutions = pd.DataFrame.from_records(solutions)
        return self.solutions

    def save_solutions_frontend(self):
        """
        This method saves the selected strategy solution of the networks to be used in the experiment frontend;
        solutions are saved in a JSON file with network id and associated list of moves
        """
        df = self.solutions

        def construct_moves(df):
            return {
                'moves': [int(df['source_node'].iloc[0])] + df["current_node"].tolist(),
                'network_id': df['network_id'].iloc[0],
                'max_level': int(df['max_level'].iloc[-1]),
                'total_reward': int(df['total_reward'].iloc[-1])
            }

        s = [
            construct_moves(df)
            for _, df in
            df.groupby(["network_id"])
        ]
        # s = (
        #     df.groupby(["network_id"])
        #     .apply(construct_moves)
        # )
        # print(s)
        # obj = s.to_dict("records")
        return json.dumps(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve networks with a rule based algorithm."
    )
    parser.add_argument(
        "-c", "--config", help="Path to the config YAML file", required=True
    )
    parser.add_argument(
        "-n", "--networks", help="Path to the networks JSON file", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output files", required=True
    )
    args = parser.parse_args()

    solve_params = load_yaml(args.config)
    seed = 43
    random.seed(seed)
    np.random.seed(seed)

    with open(args.networks) as json_file:
        networks = json.load(json_file)

    total_scores = []

    for strategy in ["myopic", "take_loss", "random"]:
        A = RuleAgent(networks, strategy, solve_params)
        A.solve()
        solutions = A.save_solutions_frontend()

        filename = f"{args.output}__{strategy}.json"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(solutions)

        total_scores.append(A.solutions.groupby(['network_id','strategy'])["reward"].sum().reset_index())

    total_scores = pd.concat(total_scores)

    total_scores = total_scores.pivot(index='network_id', columns='strategy', values='reward').reset_index()


    total_scores['valid'] = total_scores.apply(lambda x: x['myopic'] <= x['take_loss'], axis=1)

    # save valid networks
    valid_network_ids = total_scores[total_scores['valid']]['network_id'].tolist()
    valid_networks = [network for network in networks if network['network_id'] in valid_network_ids]

    with open(f"{args.output}__valid_networks.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(valid_networks))

    print(f"{len(valid_networks)} valid networks out of {len(networks)}")
