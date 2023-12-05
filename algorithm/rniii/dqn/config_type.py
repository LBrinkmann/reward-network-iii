from pydantic import BaseModel

class Config(BaseModel):
    model_type: str = "RNN"
    observation_type: str = "no_level_loss_counter"
    observation_shape: str = "concatenated"
    train_data_name: str = "networks_train.json"
    test_data_name: str = "networks_test.json"
    save_dir: str = "models"
    figures_dir: str = "figures"
    n_episodes: int = 2000
    n_networks: int = 1000
    train_batch_size: int = 100
    n_rounds: int = 8
    n_nodes: int = 10
    learning_rate: float = 1.e-3
    lr_scheduler_step: int = 500
    lr_scheduler_gamma: float = 0.8
    batch_size: int = 16
    nn_hidden_layer_size: int = 15
    memory_size: int = 500
    exploration_rate_decay: float = 0.99
    nn_update_frequency: int = 200