#!/bin/bash -l
#
#SBATCH --output=algorithm/slurm_logs/%x-%j.out
#SBATCH --job-name='reward_network_iii'
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#SBATCH --gres=gpu
#SBATCH --partition gpu

set -e

module load python/3.10
module load cuda

source .venv/bin/activate

export USE_WANDB="true"

echo "Entered environment"

python algorithm/dqn/dqn_agent.py --config algorithm/params/seed_2.yml
