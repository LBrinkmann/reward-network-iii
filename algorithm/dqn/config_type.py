from pydantic import BaseModel
from typing import List

class Config(BaseModel):
    seed: int = 0
    name: str
    tags: List[str] = []
    model_type: str = "RNN"
    observation_type: str = "no_level_loss_counter"
    observation_shape: str = "concatenated"
    train_data_name: str = "networks_train.json"
    test_data_name: str = "networks_test.json"
    exp_data_name: str = "networks_exp.json"
    solutions_dir: str = "solutions"
    save_dir: str = "models"
    figures_dir: str = "figures"
    n_episodes: int = 2000
    n_networks: int = 1000
    network_batch: int = 100
    test_period: int = 100
    n_rounds: int = 8
    n_nodes: int = 10
    learning_rate: float = 1.e-3
    lr_scheduler_step: int = 500
    lr_scheduler_gamma: float = 0.8
    batch_size: int = 16
    nn_hidden_layer_size: int = 15
    memory_size: int = 500
    exploration_rate: float = 1.0
    exploration_rate_min: float = 0.01
    exploration_rate_decay: float = 0.99
    nn_update_frequency: int = 200
    rewards: List[int] = []
