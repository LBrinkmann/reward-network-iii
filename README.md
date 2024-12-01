# Reward Networks

This repository contains the code for the project "Experimental Evidence for the Propagation and Preservation of Machine Discoveries in Human Populations" 

This repository allows to:
- Train a machine player to solve reward networks
- Run an online experiment where human participants solve reward networks task
- Analyze the data from the experiment
- Visualize the results

This repository contains both the code and the experimental data.

## Overview of Resources

### Data

The data is stored in the `data` directory. The data is structured as follows:
- `data/networks_solutions_models` contains the networks, trained neural network models, and solutions for the networks (both, from the neural networks and three prototypical heuristic strategies).
- `data/exp_raw` contains the raw data from the experiment as downloaded from the online experiment.
- `data/exp_processed` contains the processed data from the experiment, including the alignment between human and machine actions and written strategies.
- `data/abm` contains the data from the agent-based model (after running the corresponding notebook).

### Algorithm

The algorithm is implemented in the [algorithm](algorithm) directory. The algorithm trains a neural policy to solve reward networks tasks.

### Online Experiment

The online experiment is hosted on the [backend](backend) and [frontend](frontend) services. The frontend is a React application that allows participants to solve reward networks tasks. The backend is a Flask application that serves the frontend and stores the data from the experiment.

### Visualizations

The visualizations are stored in the `analysis/plots` directory. The corresponding notebooks are stored in the `analysis` directory.

### Statistical Analysis

The statistical analysis is stored in the `statistics` directory.

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

## Usage

### Training of the Machine Player

See the respective [algorithm README](algorithm/README.md) for more details.

### Run the experiment

Start the frontend and backend services using the following command:

```bash
docker compose up frontend backend
```

### Visualizations of Experimental Data, Agent-Based Model, and Algorithmic Learning Curve

See the respective [analysis README](analysis/README.md) for more details.
