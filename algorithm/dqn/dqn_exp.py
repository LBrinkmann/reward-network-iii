# This script uses a Deep Q Learning AI agent model to solve a Reward Network DAG
# used in the experiment. It returns a list of moves for each network solved in the same format used in the
# RN-III backend code.
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################

import json
import os
import glob
import torch as th
import yaml
from yaml.loader import SafeLoader
from pydantic import BaseModel
from environment_vect import Reward_Network
from logger import MetricLogger
from dqn_agent import Agent


class Config(BaseModel):
    model_type: str = "RNN"
    observation_type: str = "no_level_loss_counter"
    observation_shape: str = "concatenated"
    train_data_name: str = "networks_train.json"
    test_data_name: str = "networks_test.json"
    n_episodes: int = 1
    n_networks: int = 1000
    train_batch_size: int = 1000
    n_rounds: int = 8
    n_nodes: int = 10
    learning_rate: float = 1.e-3
    lr_scheduler_step: int = 500
    lr_scheduler_gamma: int = 0.9
    batch_size: int = 16
    nn_hidden_layer_size: int = 15
    memory_size: int = 50
    exploration_rate_decay: float = 0.99
    nn_update_frequency: int = 200


if __name__ == "__main__":

    # ---------Directory management---------------------
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    project_folder = os.path.split(os.path.split(os.getcwd())[0])[0]
    data_dir = os.path.join(project_folder, "data")
    save_dir = os.path.join(project_folder, "save")
    params_dir = os.path.join(project_folder, "params")

    # ---------Load configuration parameters-------------
    config = Config()

    # ---------Loading of the networks---------------------
    print(f"Loading experiment networks from file: {os.path.join(data_dir, config.train_data_name)}")
    # Load networks
    with open(os.path.join(data_dir, 'networks_train.json')) as json_file:
        networks_exp = json.load(json_file)
    print(f"Number of networks loaded: {len(networks_exp)}")

    # ---------Specify device (cpu or cuda)----------------
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

    # create network id - moves dictionary and save as json
    network_ids = list(map(lambda n: n["network_id"], networks_exp))

    # ---------Start analysis------------------------------
    # initialize environment(s)
    env = Reward_Network(networks_exp,
                         config.observation_shape,
                         config.observation_type,
                         config.train_batch_size,
                         DEVICE)

    logger = MetricLogger("dqn",
                          save_dir,
                          config.train_batch_size,
                          config.n_episodes,
                          config.n_nodes)

    AI_agent = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=save_dir,
        device=DEVICE,
    )

    # load model checkpoint
    checkpoint_path = glob.glob(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + f'/*.chkpt')[0]
    print(f'checkpoint file found: {checkpoint_path}')
    AI_agent.load_model(checkpoint_path)

    # solve experiment networks
    # reset env(s)
    env.reset()
    # obtain first observation of the env(s)
    obs = env.observe()

    # solve all networks at once in one go ("1 episode")
    for e in range(1):
        moves = AI_agent.solve_loop(e,
                                    "dqn",
                                    config.n_rounds,
                                    False,
                                    True,
                                    env,
                                    logger,
                                    obs)

    solutions = []
    for i in range(len(network_ids)):
        solutions.append({'network_id': network_ids[i],
                          'moves': moves[i,:].tolist()})





