# Algorithm

This project implements a reinforcement learning approach based on deep Q-learning to train a neural policy for navigating and solving reward network tasks.

### Overview
The algorithm simulates how agents explore networks by learning to maximize rewards while navigating through nodes. It mimics intuitive or heuristic decision-making by only considering immediate options without explicit planning.

### Key Features
- Trains a model-free neural policy capable of solving the reward network tasks.
- Evaluates performance against heuristic benchmarks, such as random agents and myopic strategies.
- Uses unique network environments for training and testing to ensure robustness.

## Setup

See the [Main README](../README.md) for the setup of the project.

## Generate Networks

The training of the DQN agent requires a dataset of networks. To generate the networks, run the following command:

```bash
docker compose run all python common/generate/generation.py -i config/networks_train.yml -o data/networks_train.json
```

## Train Algorithm

To train the DQN agent on the generated networks (training dataset), run the following command:

```bash
docker compose run all python algorithm/dqn/dqn_agent.py --config algorithm/params/seed_0.yml
```

## Apply Algorithm to Networks

To apply the trained DQN agent on the generated networks (experiment dataset), run the following command:

```bash
docker compose run all python algorithm/dqn/dqn_exp.py --config algorithm/params/seed_0.yml
```