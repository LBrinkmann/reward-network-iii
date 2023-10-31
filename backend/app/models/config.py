from datetime import datetime
from typing import Optional

from beanie import Document


class ExperimentSettings(Document):
    # META SETTINGS
    # whether the experiment is active
    active: bool = False
    created_at: Optional[datetime] = datetime.now()
    # Redirect URL for Prolific
    redirect_url: Optional[str] = "https://app.prolific.co/submissions/complete"

    # GENERAL SETTINGS
    # name of the experiment
    experiment_type: str = "reward-network-iii"
    # whether the experiment with the same name is to be overwritten (just for development)
    rewrite_previous_data: bool = False

    # SESSION TREE SETTINGS
    # seed for network shuffle
    seed: Optional[int]
    # number of generations (includes the seeding / first generation)
    n_generations: int = 3
    # number of AI player in seeding generation
    n_ai_players: int = 3
    # number of session per generation (after seeding generation)
    # if the seeding generation is mixed, these sessions are
    # split into two streams
    n_sessions_per_generation: int = 20
    # number of social models player can choose between
    n_advise_per_session: int = 5
    # number of complete independent replication of the experiment to be created
    n_session_tree_replications: int = 1
    # conditions of the experiment
    conditions: list = ["wo_ai", "w_ai"]

    # SESSION TRIALS SETTINGS
    n_social_learning_blocks: int = 1
    # number of trails that are passed to the next generation
    n_social_learning_networks_per_block: int = 4
    # number of practice trails
    n_practice_trials: int = 2
    # number of demonstration trials (these trails are used for social learning)
    n_demonstration_trials: int = 5
    # simulate human responses in first generation with a myopic model
    simulate_humans: bool = False
    # whether to add a repeat trial at the end of the social learning block (otherwise a try yourself trail is added)
    add_repeat_trial: bool = False
