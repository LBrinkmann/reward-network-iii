# This script uses a Deep Q Learning AI agent model to solve a Reward Network DAG
# used in the experiment. It returns a list of moves for each network solved in the same format used in the
# RN-III backend code.
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################


import yaml
import json
import os

import argparse
import glob
import torch as th
from environment_vect import Reward_Network
from logger import MetricLogger
from dqn_agent import Agent
from config_type import Config



def compute_solutions(config):
    with open(config.exp_data_name) as json_file:
        networks_exp = json.load(json_file)
    print(f"Number of networks loaded: {len(networks_exp)}")
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = Reward_Network(networks_exp, network_batch=None, config=config, device=DEVICE)

    AI_agent = Agent(
        observation_shape=env.observation_shape,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=None,
        device=DEVICE,
    )
    path = os.path.join(config.save_dir, f"{config.name}_{config.seed}.pt")
    AI_agent.load_model(path)

    # solve all networks at once in one go ("1 episode")
    for e in range(1):
        moves = AI_agent.solve_loop(
            episode=e,
            n_rounds=config.n_rounds,
            train_mode=False,
            exp_mode=True,
            env=env,
            logger=None
        )

    print('moves', moves.shape)

    solutions = []
    for i, n in enumerate(networks_exp):
        solutions.append({'network_id': n["network_id"],
                          'moves': [*moves[i,:].tolist()]})

    # save solutions as json
    with open(os.path.join(config.solutions_dir, f"{config.name}_{config.seed}.json"), 'w') as outfile:
        json.dump(solutions, outfile)


if __name__ == "__main__":

    # Load config parameter from yaml file specified in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="algorithm/params/dqn/single_run_v2.yml", help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(**config)
    compute_solutions(config)

