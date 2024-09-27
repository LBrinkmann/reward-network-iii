# Reward Network III

Brief description of what the project does and its purpose.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Docker

Network generation, training of the algorithm, and running the backend can be done using Docker. To build the Docker image, run the following command:

```bash
docker compose build all
```

### Python

For running the descriptive analysis of the results, the following setup is required:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -e ".[viz]"
```

Or us the following command to install all dependencies:

```bash
pip install -e ".[viz,dev,backend,train]"
```

### R

For the statistical analysis of the results, the following setup is required:

```bash

```

## Usage

### Train Algorithm

To train the DQN agent on the generated networks (training dataset), run the following command:

```bash
docker compose run all python algorithm/dqn/dqn_agent.py --config algorithm/params/seed_0.yml
```

To apply the trained DQN agent on the generated networks (experiment dataset), run the following command:

```bash
docker compose run all python algorithm/dqn/dqn_exp.py --config algorithm/params/seed_0.yml
```

### Compute Machine-Human Agreement

To compute the agreement between human and machine actions, run the following Notebook:

[algorithm/compute_alignment.ipynb](algorithm/compute_alignment.ipynb)

The results are stored in the following directory:

`analysis/data/experiment/processed/moves_w_alignment.csv`
