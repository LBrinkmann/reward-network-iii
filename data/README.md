# Data

## Experimental Data

### Raw Data

In the folder `exp_raw` you can find the raw data for the experimental task as downloaded from the experimental interface.

### Processed Data

In the folder `exp_processed` you can find the processed data for the experimental task. This includes:
- 'moves_w_alignment.csv' a table with each row signifying a move in the game.
- 'player.csv' a table with each row signifying a player in the game.

#### Moves

The table `moves_w_alignment.csv` contains the following columns:
- `session_id`: the session id
- `session_name`: a descriptive name for the session
- `replication_idx`: the replication index
- `condition`: the experimental condition of the session
- `generation`: the generation within the experimental condition
- `within_generation_idx`: the index of the session within the generation (0 - 7)
- `started_at`: the time the session started
- `time_spend`: the time spend in the session
- `expired`: whether the session expired (expired sessions are excluded from the analysis)
- `replaced`: whether the session was replaced (expired sessions are replaced by new sessions)
- `ai_player`: whether the player was an AI player
- `simulated_subject`: whether the player was a simulated subject (debugging purposes)
- `advisor`: the advisor (i.e. selected demonstrator) of the player
- `player_score`: the aggregated score of the player
- `move_idx`: the index of the move within the trial
- `source_num`: the source node of the move
- `target_num`: the target node of the move
- `reward`: the reward of the move
- `level`: the level of the source node
- `correct_repeat`: whether the move was a correct repeat (applies to 'repeat' trials only)
- `myopic`: whether the move was myopic (i.e. the player took a move with the highest immediate reward)
- `optimal`: whether the move was optimal (i.e. the player took a move with the highest cumulative reward)
- `large_loss_taken`: whether the player took a large loss
- `trial_id`: the trial id
- `trial_type`: the trial type (demonstration, observation, try_yourself, repeat, individual)
- `network_id`: the network id
- `solution_total_score`: the total score of the full solution of that player in that trial
- `session_trial_id`: unique identifier for the session + trial
- `human_machine_match`: whether move of the player matched the move of the machine player

#### Player

The table `player.csv` contains the following columns:
- `session_id`: the session id
- `session_name`: a descriptive name for the session
- `replication_idx`: the replication index
- `condition`: the experimental condition of the session
- `generation`: the generation within the experimental condition
- `within_generation_idx`: the index of the session within the generation (0 - 7)
- `started_at`: the time the session started
- `time_spend`: the time spend in the session
- `expired`: whether the session expired (expired sessions are excluded from the analysis)
- `replaced`: whether the session was replaced (expired sessions are replaced by new sessions)
- `ai_player`: whether the player was an AI player
- `simulated_subject`: whether the player was a simulated subject (debugging purposes)
- `advisor`: the advisor (i.e. selected demonstrator) of the player
- `player_score`: the aggregated score of the player


### Coded Strategies

In the folder `exp_strategies_coded` you can find the coded strategies for the experimental task. This includes:
- `ratings_full.csv` a table with the ratings for each strategy from each rater.
- `coding_id_map.csv` a table with the mapping of the coding ids to session ids.
- `coded_strategies.csv` a table with the coded strategies for each session.

#### Coded Strategies

The table `coded_strategies.csv` contains the following columns:
- `session_id`: the session id
- `replication_idx`: the replication index
- `trial_id`: the trial id
- `coding_id`: the coding id (used during the coding process)
- `condition`: the experimental condition of the session
- `generation`: the generation within the experimental condition
- `written_strategy_idx`: the index of the written strategy
- `text`: the text of the written strategy
- `loss_strategy`: whether the strategy was coded as a loss strategy (0 = no, 1 = yes)

## Networks, Solutions and Models

In the folder `networks_solutions_models` you can find:
- 'networks_*.json' files with define the experimental task. This involves the nodes (incl. node level) and the edges (incl. rewards).
- 'solutions_*.json' the solutions for each network following different heuristic strategies (myopic, random, take losses).
- 'model/*.pt' contain the model weights for the machine player.
- 'model/*.csv' contains training metrics recorded during training of the machine player.
- 'machine_solution/*.json' the solutions for each network following the machine learning strategies.