# reward-network-iii-analysis

This repository contains scripts to parse and analyze data from the Reward Networks III experiment.

## Repository organization
The repository is organized as follows:
* Python scripts `donwload_results.py` and `parse.py` are scripts used to download rawdata from the experiment database and to create base tables for the analysis, respectively
* the folder `data` contains rawdata and parsed data, as well as plot-specific tables
* the folder `analysis` contains figures and R scripts used in the analysis 
* the folder `models` contains pydantic classes used in parsing the data


## Set up the repository

### Virtual environment

```bash
python3 -m venv venv
 
source venv/bin/activate

python3 -m pip install --upgrade pip

```

### Setup .env

You need to create a .env file in the root directory of the project. The .env
file should contain the following variables:

```dotenv
BACKEND_USER=<your backend user>
BACKEND_PASSWORD=<your backend password>
BACKEND_URL=<your backend url>
```

### Downloading the data

```bash

python3 download_data.py

```

### Create data tables

```bash

python3 parse.py

```

## Data tables details

### Player (base table)

| Column Name    | Type        | Description                               |
|----------------|-------------|-------------------------------------------|
| sessionId      | pydantic id | session identifier                        |
| isAI           | bool        | indicates whether session is bot or human |
| condition      | str         | lineage condition of the session          |
| experimentName | str         | name of experiment                        |

### Trial (base table)

| Column Name        | Type        | Description                                                                      |
|--------------------|-------------|----------------------------------------------------------------------------------|
| sessionId          | pydantic id | session identifier                                                               |
| generation         | int         | generation index                                                                 |
| trialType          | str         | trial type                                                                       |
| trialIdx           | int         | trial index                                                                      |
| parentSessionId    | pydantic id | parent session identifier (applies to social learning trials)                    |
| parentSession_isAI | bool        | identifies if parent session is bot or human (applies to social learning trials) |

### Network (base table)

| Column Name          | Type        | Description                                            |
|----------------------|-------------|--------------------------------------------------------|
| networkId            | pydantic id | reward network identifier                              |
| expectedMyopicReward | int         | score from myopic agent                                |
| expectedAiReward     | int         | score from 'take first loss, then behave myopic' agent |

### Moves (base table)

| Column Name   | Type        | Description                               |
|---------------|-------------|-------------------------------------------|
| sessionId     | pydantic id | session identifier                        |
| trialId       | int         | trial index                               |
| networkId     | pydantic id | reward network identifier                 |
| sourceLevel   | int         | level of current node                     |
| targetLevel   | int         | level of node reached after taking step   |
| reward        | int         | reward obtained in step                   |
| step          | int         | step number                               |
| matchesMyopic | bool        | is this step the same as myopic solution? |
| matchesAI     | bool        | is this step the same as ai solution?     |

### Written strategy (base table)

| Column Name   | Type        | Description                        |
|---------------|-------------|------------------------------------|
| sessionId     | pydantic id | session identifier                 |
| condition     | str         | lineage condition of the session   |
| strategyText  | str         | written strategy text              |
| strategyScore | int         | score of the written strategy text |

### Post survey (base table)

| Column Name           | Type        | Description                                                  |
|-----------------------|-------------|--------------------------------------------------------------|
| sessionId             | pydantic id | session identifier                                           |
| strategy              | str         | written strategy text (in post survey textbox)               |
| task_explanation      | int         | (1 to 5, higher is clearer)                                  |
| difficulty_rating     | int         | (1 to 5, higher is more difficult)                           |
| time_limit_sufficient | int         | (1 to 5, higher is too much time)                            |
| color_of_arrows       | int         | (1 to 5, higher is hard to distinguish)                      |
| additional_comments   | str         | comments from final post survey suggestions/comments textbox |
