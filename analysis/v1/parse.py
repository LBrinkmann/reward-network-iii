# This script parses downloaded data in JSON file using Pydantic models
# and creates data tables 
# Project: Reward Networks III
######################################

from models.session import Session
from models.network import Network
from models.trial import Trial
from dotenv import load_dotenv
from pydantic import ValidationError
import json
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns


class Parser:

    def __init__(self, raw_data_path, experiment_name):
        """
        Initializes parser object

        Args:
            raw_data_path (str): path to raw experiment specific data json file
            experiment_name (str): name of the experiment
        """

        # assert
        assert os.path.exists(raw_data_path), f'data file not found!'

        self.raw_data_path = raw_data_path
        self.experiment_name = experiment_name
        print(f"EXPERIMENT NAME: {self.experiment_name}")
        self.n_steps = 8
        self.base_table_names = ['moves', 'trial', 'network', 'player', 'written_strategy']

        # pilot 1-2
        # self.solution_filenames = {'AI': 'solution_moves_take_first_loss_viz.json',
        #                           'Myopic': 'solution_moves_highest_payoff_viz.json'}
        # pilot 3 and onwards
        self.solution_filenames = {'AI': 'solutions_loss.json',
                                   'Myopic': 'solutions_myopic.json'}

    def open_data_file(self):
        """
        Opens the data of the specified experiment, both subjects 
        (no profilic id info!) and sessions
        """
        self.session_isAI_dict = {}

        print(self.raw_data_path)
        # Opening JSON file (sessions)
        with open(self.raw_data_path) as json_file:
            sessions = json.load(json_file)
        try:
            # validate data using pydantic Session model and create list of sessions
            self.sessions_list = [Session(**s) for s in sessions]
            for s in self.sessions_list:
                self.session_isAI_dict[s.id] = s.ai_player

                for t in s.trials:
                    if t.trial_type is not None and t.solution is not None:
                        # if t.solution.score < (-400):
                        print(s.id, t.trial_type, t.solution.moves, t.solution.score)

        except ValidationError as e:
            print(e)

    @staticmethod
    def get_network_info(network_obj):
        """
        This method returns two dictionaries, one with the level of each node
        and the other with the reward associated to each (node,node) edge present
        in the network

        Args:
            network_obj (pydantic Network): a Network object, containing nodes and edges info
        """

        level_info = {n.node_num: n.level for n in network_obj.nodes}
        reward_info = {(e.source_num, e.target_num): e.reward for e in network_obj.edges}
        return level_info, reward_info

    def network_compare_solution(self, starting_node: int, network_id, trial_id, trial_moves, level_info, reward_info,
                                 return_all_info=False):
        """
        Load the solution for different strategies of a network given its ID

        Args:
            starting_node (int): starting node
            network_id (pydantic id): id of the network object
            trial_id (_type_): id of the trial of interest
            trial_moves (list):list of moves in form of int e.g. [0,3,4,5]
            level_info (dict): a dict mapping node number to its level
            reward_info (dict): a dict mapping (source node, target node) tuples to its reward
            return_all_info (bool) (optional): flag to return all info calculated in the method
        """

        # load solutions
        sdir = os.path.join(ddir, 'solutions', self.experiment_name[:-1])
        print(f'Fetch solutions from directory: {sdir}')

        # load solutions
        with open(os.path.join(sdir, self.solution_filenames['AI'])) as ai_file:
            ai_solutions = json.load(ai_file)
        with open(os.path.join(sdir, self.solution_filenames['Myopic'])) as myopic_file:
            myopic_solutions = json.load(myopic_file)

        # get solution moves (myopic) 
        myopic_solution = [a['moves'] for a in myopic_solutions if a['network_id'] == network_id][0]
        # fix the first node in solution to be the actual starting node and not always 0 (relevant for pilot 2)
        myopic_solution[0] = starting_node

        # does it match myopic?
        # matches_myopic = [x == y for (x, y) in zip(trial_moves, myopic_solution)]
        trial_move_matches_myopic = [x == y for (x, y) in zip(trial_moves, myopic_solution)]
        matches_myopic = [trial_move_matches_myopic[i] == trial_move_matches_myopic[i + 1] == True
                          for i in range(len(trial_move_matches_myopic) - 1)]
        if len(trial_moves) != (self.n_steps + 1):
            matches_myopic = matches_myopic + [False] * (8 - len(matches_myopic))

        # get solution moves (ai) 
        ai_solution = [a['moves'] for a in ai_solutions if a['network_id'] == network_id][0]
        # fix the first node in solution to be the actual starting node and not always 0 (relevant for pilot 2)
        ai_solution[0] = starting_node
        # does it match ai?
        # matches_ai = [x == y for (x, y) in zip(trial_moves, ai_solution)]
        trial_move_matches_ai = [x == y for (x, y) in zip(trial_moves, ai_solution)]
        matches_ai = [trial_move_matches_ai[i] == trial_move_matches_ai[i + 1] == True
                      for i in range(len(trial_move_matches_ai) - 1)]
        if len(trial_moves) != (self.n_steps + 1):
            matches_ai = matches_ai + [False] * (8 - len(matches_ai))

        # translate list of moves into list of level and reward progression
        sourceLevel = [level_info[trial_moves[n]] for n in range(len(trial_moves) - 1)]
        targetLevel = [level_info[trial_moves[n + 1]] for n in range(len(trial_moves) - 1)]
        reward = [reward_info[(trial_moves[n], trial_moves[n + 1])] for n in range(len(trial_moves) - 1)]
        ai_reward = sum([reward_info[(ai_solution[n], ai_solution[n + 1])] for n in range(len(ai_solution) - 1)])
        myopic_reward = sum(
            [reward_info[(myopic_solution[n], myopic_solution[n + 1])] for n in range(len(myopic_solution) - 1)])

        print('source level', sourceLevel, f'({len(sourceLevel)})')
        print('target level', targetLevel, f'({len(targetLevel)})')
        print('reward list', reward, f'({len(reward)})')
        print(f'return_all_info is {return_all_info}')

        # record missing steps info
        # ismissing_bool = [False] * len(reward)
        # if len(trial_moves) != (self.n_steps+1):
        #    ismissing_bool = ismissing_bool + [True] * (8 - len(reward))

        # adjust reward logs depending on presence of missing steps
        if len(sourceLevel) != 0 and len(targetLevel) != 0:

            ismissing_bool = [False] * len(reward)
            if len(reward) != self.n_steps:
                ismissing_bool = ismissing_bool + [True] * (8 - len(reward))
                sourceLevel = sourceLevel + [targetLevel[-1]] * (8 - len(reward))
                targetLevel = targetLevel + [targetLevel[-1]] * (8 - len(reward))
                reward = reward + [-50] * (8 - len(reward))
                max_level_reached = max(sourceLevel + targetLevel)
            else:
                max_level_reached = max(sourceLevel + targetLevel)

        else:
            sourceLevel = [0] * self.n_steps
            targetLevel = [0] * self.n_steps
            max_level_reached = 0
            reward = [-50] * self.n_steps
            ismissing_bool = [True] * self.n_steps

        print('any missings?', ismissing_bool)
        print('matches myopic?', matches_myopic)
        print('matches ai?', matches_ai)
        print('\n')

        # create dataframe
        d = pd.DataFrame({'trialID': [trial_id] * self.n_steps,
                          'networkId': [network_id] * self.n_steps,
                          'sourceLevel': sourceLevel,
                          'targetLevel': targetLevel,
                          'reward': reward,
                          'step': [i + 1 for i in range(self.n_steps)],
                          'isMissing': ismissing_bool,
                          'matchesMyopic': matches_myopic,
                          'matchesAI': matches_ai})

        # # create dataframe
        # d = pd.DataFrame({'trialID': [trial_id] * len(reward),
        #                   'networkId': [network_id] * len(reward),
        #                   'sourceLevel': sourceLevel,
        #                   'targetLevel': targetLevel,
        #                   'reward': reward,
        #                   'step': [i + 1 for i in range(len(reward))],
        #                   'matchesMyopic': matches_myopic[:len(reward)],
        #                   'matchesAI': matches_ai[:len(reward)]})
        if return_all_info:
            return d, ai_reward, myopic_reward, max_level_reached
        else:
            return d

    def calculate_session_trial_metrics(self, session):
        """
        Given a session this method returns for each trial type of interest
        metrics like average score (as in reward) and maximum level reached

        Args:
            session (Session pydantic model): a session object containing id, ai flag, list of trials,...
        """
        # scores_trialType = {'social_learning_tryyourself': [], 'individual': [], 'demonstration': []}
        # maxLevel_trialType = {'social_learning_tryyourself': [], 'individual': [], 'demonstration': []}
        # net_ids = {'social_learning_tryyourself': [], 'individual': [], 'demonstration': []}
        # trial_ids = {'social_learning_tryyourself': [], 'individual': [], 'demonstration': []}

        scores_trialType = {'try_yourself': [], 'individual': [], 'demonstration': []}
        maxLevel_trialType = {'try_yourself': [], 'individual': [], 'demonstration': []}
        net_ids = {'try_yourself': [], 'individual': [], 'demonstration': []}
        trial_ids = {'try_yourself': [], 'individual': [], 'demonstration': []}

        for t in session.trials:
            # if t.trial_type in ['social_learning', 'individual', 'demonstration'] and t.solution.score != -100000:
            if t.trial_type in ['try_yourself', 'individual', 'demonstration'] and t.solution.score != -100000:

                # exclude the two trials with wrong frontend solutions using network id
                if t.network is not None and t.network.network_id not in ['6384fdeb6fb4a4038d4ce96d4de6348c',
                                                                          '6ba707ecfd7766fb464cf738dbc643e3']:
                    # max level reached part 
                    level_info, _ = self.get_network_info(t.network)
                    max_level = max([level_info[n] for n in t.solution.moves])

                    # if t.trial_type == 'social_learning' and t.social_learning_type == 'tryyourself':
                    if t.trial_type == 'try_yourself':

                        # maxLevel_trialType[t.trial_type + '_' + t.social_learning_type].append(max_level)
                        # scores_trialType[t.trial_type + '_' + t.social_learning_type].append(t.solution.score)
                        # net_ids[t.trial_type + '_' + t.social_learning_type].append(t.network.network_id)
                        # trial_ids[t.trial_type + '_' + t.social_learning_type].append(t.id)

                        maxLevel_trialType[t.trial_type].append(max_level)
                        scores_trialType[t.trial_type].append(t.solution.score)
                        net_ids[t.trial_type].append(t.network.network_id)
                        trial_ids[t.trial_type].append(t.id)

                    elif t.trial_type in ['individual', 'demonstration']:

                        maxLevel_trialType[t.trial_type].append(max_level)
                        scores_trialType[t.trial_type].append(t.solution.score)
                        net_ids[t.trial_type].append(t.network.network_id)
                        trial_ids[t.trial_type].append(t.id)

        return maxLevel_trialType, scores_trialType, net_ids, trial_ids

    #### base tables #####
    def create_base_table_moves(self):
        """
        Generates base moves table with columns
        - sessionId (id)
        - trialId (id)
        - networkId (id)
        - sourceLevel (int)
        - targetLevel (int)
        - reward (int) 
        - step (int)
        - matchesMyopic (bool?)
        - matchesAI (bool?)
        """

        data = []
        self.moves_table_columns = ['sessionId', 'trialId', 'networkId', 'sourceLevel', 'targetLevel',
                                    'reward', 'step', 'isMissing', 'matchesMyopic', 'matchesAI']

        for s in self.sessions_list:
            for t in s.trials:

                # exclude the two trials with wrong frontend solutions using network id
                # + include only demonstration, individual and social learning trials
                # if t.network is not None and t.network.network_id not in ['6384fdeb6fb4a4038d4ce96d4de6348c','6ba707ecfd7766fb464cf738dbc643e3'] and t.trial_type in ['social_learning','individual','demonstration']:
                if t.network is not None and t.network.network_id not in ['6384fdeb6fb4a4038d4ce96d4de6348c',
                                                                          '6ba707ecfd7766fb464cf738dbc643e3'] and (
                        t.trial_type in ['individual', 'demonstration'] or (t.trial_type == 'try_yourself')):
                    # (t.trial_type == 'social_learning' and t.social_learning_type == 'tryyourself')):

                    print(t.id)
                    level_info, reward_info = self.get_network_info(t.network)
                    dataframe = self.network_compare_solution(t.network.starting_node,
                                                              t.network.network_id,
                                                              t.id,
                                                              t.solution.moves,
                                                              level_info,
                                                              reward_info)
                    dataframe.insert(0, 'sessionId', '')
                    dataframe['sessionId'] = s.id
                    data.append(dataframe)

        # make dataframe and save it as csv
        moves = pd.concat(data, ignore_index=True)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        moves.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'moves.csv'), sep=',')

    def create_base_table_trial(self):
        """
        Generates base trial table with columns:
        - sessionId (id)
        - generation (int)
        - trialType (str)
        - trialIdx (int)
        - parentSessionId (id)
        - parentSession_isAI (bool)
        """

        data = []
        self.trial_table_columns = ['sessionId', 'generation', 'trialType', 'trialIdx', 'parentSessionId',
                                    'parentSession_isAI']

        for s in self.sessions_list:
            for t in s.trials:
                # if t.trial_type not in ['social_learning']:
                if t.trial_type not in ['observation', 'repeat', 'try_yourself']:
                    data.append((s.id, s.generation, t.trial_type, t.id, None, None))
                else:
                    data.append((s.id,
                                 s.generation,
                                 t.trial_type,  # t.trial_type + '_' + t.social_learning_type,
                                 t.id,
                                 t.advisor.advisor_id,
                                 self.session_isAI_dict[t.advisor.advisor_id]))

        # make dataframe and save it as csv
        trial = pd.DataFrame(data, columns=self.trial_table_columns)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        trial.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'trial.csv'), sep=',')

    def create_base_table_network(self):
        """
        Generates base table network with columns
        - networkId (id)
        - expectedAIReward (int)
        - expectedMyopicReward (int)
        - expectedRandomReward (int) TODO
        """
        data = []
        self.network_table_columns = ['networkId', 'expectedAIReward', 'expectedMyopicReward']
        for s in self.sessions_list:
            for t in s.trials:
                if t.network is not None:
                    if t.network.network_id in ['6384fdeb6fb4a4038d4ce96d4de6348c', '6ba707ecfd7766fb464cf738dbc643e3']:
                        print(f'session id {s.id} - network id {t.network.network_id}, {t.trial_type}, CHECK')
                        level_info, reward_info = self.get_network_info(t.network)
                        print(f'network id {t.network.network_id} with solution {t.solution.moves}')
                        for k, v in reward_info.items():
                            print(k, v)
                    else:
                        print(f'session id {s.id} - network id {t.network.network_id}, {t.trial_type}')
                    level_info, reward_info = self.get_network_info(t.network)
                    # try to skip the network with wrong solution
                    if t.network.network_id not in ['6384fdeb6fb4a4038d4ce96d4de6348c',
                                                    '6ba707ecfd7766fb464cf738dbc643e3']:
                        d, expected_ai, expected_myopic, max_level = self.network_compare_solution(
                            t.network.starting_node,
                            t.network.network_id,
                            t.id,
                            t.solution.moves,
                            level_info,
                            reward_info,
                            return_all_info=True)
                    data.append((t.network.network_id, expected_ai, expected_myopic))

        # make dataframe and save it as csv
        network = pd.DataFrame(data, columns=self.network_table_columns)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        network.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'network.csv'), sep=',')

    def create_base_table_player(self):
        """
        Generate base table player, which includes session id, if session is from AI or not,
        the experimental condition and name of the experiment player participated in
        Columns:
        - sessionId (id)
        - isAI (bool)
        - condition (str)
        - experimentName (str)
        """

        condition_dict = {'A': 'human_lineage', 'a': 'Human_lineage',
                          'B': 'AI_lineage', 'b': 'AI_lineage',
                          'C': 'Myopic_lineage', 'c': 'Myopic_lineage'}
        self.player_table_columns = ['sessionId', 'isAI', 'condition', 'ExperimentName']
        p = []

        for s in self.sessions_list:
            p.append((s.id,
                      s.ai_player,
                      condition_dict[self.experiment_name[-1]],
                      self.experiment_name))

        # make dataframe and save it as csv
        player = pd.DataFrame(p, columns=self.player_table_columns)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        player.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'player.csv'), sep=',')

    def create_written_strategy_table(self):
        """
        Creates a table that for each session ID (for the written strategy trial)
        returns corresponding written strategy text.
        Note: only sessions from humans are included here!

        Table with columns:
        - sessionID (id)
        - condition (str)
        - strategyText (str)
        - strategyScore (int) TODO
        """

        condition_dict = {'A': 'human_lineage', 'a': 'Human_lineage',
                          'B': 'AI_lineage', 'b': 'AI_lineage',
                          'C': 'Myopic_lineage', 'c': 'Myopic_lineage'}
        self.written_strategy_table_columns = ['sessionId', 'condition', 'strategyText', 'strategyScore']
        data = []

        for s in self.sessions_list:
            if s.subject_id is not None:  # include only written strategy from humans
                for t in s.trials:
                    if t.trial_type == 'written_strategy':
                        data.append((s.id,
                                     condition_dict[self.experiment_name[-1]],
                                     t.written_strategy.strategy,
                                     'TODO'))

        # make dataframe and save it as csv
        written_strategy = pd.DataFrame(data, columns=self.written_strategy_table_columns)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        written_strategy.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'written_strategy.csv'), sep=',')

    def create_advisor_table(self):
        """
        Creates a table that summarizes the advisors type (myopic or loss)
        and the scores - relevant for pilots where advisors are present (e.g. not applicable in pilots of type A)
        """
        advisors_dict = {}

        for s in self.sessions_list:
            for t in s.trials:
                if t.trial_type == 'social_learning_selection':
                    for a in range(len(t.advisor_selection.advisor_ids)):
                        advisors_dict[str(t.advisor_selection.advisor_ids[a])] = t.advisor_selection.scores[a]

        advisor_df = pd.DataFrame(advisors_dict.items(), columns=['advisorId', 'advisorScore'])
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        advisor_df.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'advisor.csv'), sep=',')

    def create_post_survey_table(self):
        """
        Creats a table collecting post survey questionnaire answers and additional
        comments for each session id, columns are:

        - sessionId
        - strategy
        - task_explanation
        - difficulty_rating
        - time_limit_sufficient
        - arrows_color
        - additional_comments
        """
        data = []
        self.post_survey_table_columns = ['sessionId', 'task_explanation',
                                          'difficulty_rating', 'time_limit_sufficient', 'arrows_color',
                                          'additional_comments']

        for s in self.sessions_list:
            for t in s.trials:
                if t.trial_type == 'post_survey':
                    # pilot 1-2
                    # data.append([s.id,
                    #              t.post_survey.questions['0'],
                    #              int(t.post_survey.questions['1']),
                    #              int(t.post_survey.questions['2']),
                    #              int(t.post_survey.questions['3']),
                    #              int(t.post_survey.questions['4']),
                    #              t.post_survey.questions['5']])

                    # pilot 3 and onwards
                    data.append([s.id,
                                 int(t.post_survey.questions['0']),
                                 int(t.post_survey.questions['1']),
                                 int(t.post_survey.questions['2']),
                                 int(t.post_survey.questions['3']),
                                 t.post_survey.questions['4']])

        # make dataframe and save it as csv
        post_survey = pd.DataFrame(data, columns=self.post_survey_table_columns)
        if not os.path.exists(os.path.join(ddir, 'final', self.experiment_name)):
            os.makedirs(os.path.join(ddir, 'final', self.experiment_name))
        post_survey.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'post_survey.csv'), sep=',')

    ###### examine missings #######
    def examine_missing(self):
        """
        This method returns for each experiment the counts of the number of moves used in
        each trial with trial type (social_learning_tryyourself,individual,demonstration)

        Returns:
            _type_: _description_
        """
        # print(self.experiment_name)
        # for s in self.sessions_list:
        #     for t in s.trials:
        #         if t.trial_type in ['social_learning','individual','demonstration']:
        #             if t.trial_type =='social_learning':
        #                 print(t.trial_type,' ',t.social_learning_type,' ',t.solution.moves)
        #             else:
        #                 print(t.trial_type,' ',t.solution.moves)

        player = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'player.csv'), sep=',')
        moves = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'moves.csv'), sep=',')
        trial = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'trial.csv'), sep=',')
        moves_count = moves.groupby(by=["sessionId", 'trialID'])['step'].count().reset_index()
        moves_count = moves_count.merge(player, how='inner', on='sessionId')
        moves_count = moves_count.merge(trial[['sessionId', 'generation', 'trialType', 'trialIdx']],
                                        how='inner',
                                        left_on=['sessionId', 'trialID'],
                                        right_on=['sessionId', 'trialIdx'])

        moves_count.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'moves_count.csv'), sep=',')
        print(f'length of {len(moves_count)} ({self.experiment_name})')
        print(moves_count[['step', 'isAI']].value_counts(normalize=True))
        print(moves_count[['trialID', 'trialType', 'step', 'isAI']].value_counts(normalize=True))

        df = moves_count[['trialID', 'trialType', 'step', 'isAI']].value_counts(normalize=True).to_frame().reset_index()
        df.columns = ['trialId', 'trialType', 'step', 'isAI', 'count']

        # plot to see where in the trial order we get the most missings
        plot_hue = df['trialType'].astype(str) + ', ' + df['isAI'].astype(str)
        fig = sns.catplot(data=df, x='trialId', y='count', hue='step', col='trialType', row='isAI', kind='bar')
        fig.savefig(os.path.join('/Users/bonati/Desktop', 'where_missings.png'), format='png', dpi=300)

    ###### aggreagted tables #######
    def create_aggregated_tables(self):
        """
        _summary_
        """
        # assert tests
        assert os.path.exists(os.path.join(ddir, 'final', self.experiment_name, 'player.csv')), \
            f'base table PLAYER for experiment {self.experiment_name} not found!'
        assert os.path.exists(os.path.join(ddir, 'final', self.experiment_name, 'trial.csv')), \
            f'base table TRIAL for experiment {self.experiment_name} not found!'
        assert os.path.exists(os.path.join(ddir, 'final', self.experiment_name, 'moves.csv')), \
            f'base table MOVES for experiment {self.experiment_name} not found!'
        assert os.path.exists(os.path.join(ddir, 'final', self.experiment_name, 'network.csv')), \
            f'base table NETWORK for experiment {self.experiment_name} not found!'

        # load tables
        player = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'player.csv'), sep=',')
        moves = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'moves.csv'), sep=',')
        trial = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'trial.csv'), sep=',')
        network = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'network.csv'), sep=',')

        # PLAYER AGGREGATED
        # first join player and trial
        player_trial = trial.merge(player, how='inner', on='sessionId')
        # get number of ai parents selected
        n_ai_selected = player_trial.groupby(by=['sessionId', 'generation', 'condition']).apply(
            lambda x: x['parentSession_isAI'].count() / 6).reset_index(name='nAIParentsSelected')

        # get average reward obtained in the different trial types (social learning, individual, demonstration)
        player_trial_moves = moves.merge(trial.merge(player, how='inner', on='sessionId'), how='inner', on='sessionId')
        player_trial_moves = player_trial_moves[player_trial_moves['trialType'].isin(['individual',
                                                                                      'demonstration',
                                                                                      'social_learning_tryyourself'])]
        a = player_trial_moves.groupby(by=['sessionId', 'generation', 'trialID', 'trialType', 'networkId']).apply(
            lambda x: x['reward'].sum()).reset_index(name='final_reward').groupby(
            by=['sessionId', 'generation', 'trialType'])['final_reward'].agg(pd.Series.tolist).reset_index(
            name='all_rewards')
        a['avgReward'] = a['all_rewards'].apply(np.mean)

        player_aggr = a.merge(n_ai_selected, how='inner', on='sessionId')
        print(player_aggr.columns)
        player_aggr['experimentName'] = self.experiment_name
        player_aggr['avgReward'] = player_aggr['all_rewards'].apply(np.mean)
        player_aggr = player_aggr[['sessionId', 'experimentName', 'generation_x', 'condition', 'trialType', 'avgReward',
                                   'nAIParentsSelected']]
        player_aggr.to_csv(os.path.join(ddir, 'final', self.experiment_name, 'player_aggr.csv'), sep=',')

    def create_data_for_plots(self):
        """
        Thsi function creates for each pilot or experiment iteration data files that
        aggregate information
        """
        # open data files
        all_sessions = {}
        for folder in glob.glob(f"{ddir}/raw/*/", recursive=True):
            if folder[-3] == '5':  # substitute number with pilot number
                with open(glob.glob(os.path.join(folder + f'sessions_*.json'))[0]) as json_file:
                    sessions = json.load(json_file)
                all_sessions[os.path.basename(folder[:-1])] = [Session(**s) for s in sessions]

        all_session_isAI = {s.id: s.ai_player for k, v in all_sessions.items() for s in v}

        trial_durations = []
        durations = []
        scores = []
        # types_include = ['social_learning', 'individual', 'demonstration', 'written_strategy']
        types_include = ['try_yourself', 'individual', 'demonstration', 'written_strategy']

        for k, sessions in all_sessions.items():
            for s in sessions:
                # if s.subject_id is not None: # exclude ai for these plots
                level_metrics, score_metrics, net_ids, trial_ids = self.calculate_session_trial_metrics(s)
                for k2, v in score_metrics.items():
                    # scores.append([s.subject_id,k,s.generation,k2,score_metrics[k2],level_metrics[k2]])

                    if len(v) > 1:
                        for i in range(len(v)):
                            scores.append(
                                [s.id, all_session_isAI[s.id], k, s.generation, trial_ids[k2][i], k2, net_ids[k2][i],
                                 score_metrics[k2][i], int(level_metrics[k2][i])])
                    else:
                        scores.append([s.id, all_session_isAI[s.id], k, s.generation, None, k2, None, None, None])

                if s.time_spent.total_seconds() != 0:  # ai trials will have duration 0, exclude
                    durations.append([k, s.generation, s.time_spent.total_seconds() / 60])

                    for t in s.trials:
                        if t.trial_type in types_include and t.started_at is not None:
                            if t.trial_type == 'try_yourself':
                                trial_durations.append([k, s.subject_id,
                                                        t.id,
                                                        t.trial_type,
                                                        (t.finished_at - t.started_at).seconds])
                            #
                            # if t.trial_type == 'social_learning':
                            #     if t.social_learning_type == "tryyourself":
                            #         trial_durations.append([k, s.subject_id,
                            #                                 t.id,
                            #                                 t.trial_type + '_' + t.social_learning_type,
                            #                                 (t.finished_at - t.started_at).seconds])
                            else:
                                trial_durations.append([k, s.subject_id,
                                                        t.id,
                                                        t.trial_type,
                                                        (t.finished_at - t.started_at).seconds])

        # save the dataframes that are used for plotting
        if not os.path.exists(os.path.join(ddir, 'for_plots', self.experiment_name[:-1])):
            os.makedirs(os.path.join(ddir, 'for_plots', self.experiment_name[:-1]))
        sessions_time = pd.DataFrame(durations, columns=['expName', 'generation', 'duration'])
        sessions_time.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'sessions_time.csv'), sep=',')

        trials_time = pd.DataFrame(trial_durations,
                                   columns=['expName', 'sessionId', 'trialId', 'trialType', 'duration'])
        trials_time.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'trials_time.csv'), sep=',')

        scores_df = pd.DataFrame(scores, columns=['sessionId', 'isAI', 'expName', 'generation', 'trialIdx', 'trialType',
                                                  'networkId', 'score', 'maxLevelReached'])

        # for scores add also myopic and AI scores
        # TODO adapt for pilot 2
        # networkA = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-1A', 'network.csv'), sep=',')
        # networkB = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-1B', 'network.csv'), sep=',')

        if self.experiment_name.startswith('rn-iii-pilot-2'):
            networkA = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-2A', 'network.csv'), sep=',')
            networkB = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-2B', 'network.csv'), sep=',')
            network = pd.concat([networkA, networkB], ignore_index=True)
        elif self.experiment_name.startswith('rn-iii-pilot-3'):
            network = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-3B', 'network.csv'), sep=',')
        elif self.experiment_name.startswith('rn-iii-pilot-4'):
            network = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-4b', 'network.csv'), sep=',')
        elif self.experiment_name.startswith('rn-iii-pilot-5'):
            # network = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5b', 'network.csv'), sep=',')
            networkA = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5a', 'network.csv'), sep=',')
            networkB = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5b', 'network.csv'), sep=',')
            networkC = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5c', 'network.csv'), sep=',')
            network = pd.concat([networkA, networkB, networkC], ignore_index=True)

        network_dict_myopic = dict(zip(network.networkId, network.expectedMyopicReward))
        network_dict_ai = dict(zip(network.networkId, network.expectedAIReward))

        scores_df['myopic_score'] = scores_df['networkId'].map(network_dict_myopic)
        scores_df['ai_score'] = scores_df['networkId'].map(network_dict_ai)
        scores_df.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'scores.csv'), sep=',')

        # post survey dataframe
        if self.experiment_name.startswith('rn-iii-pilot-2'):
            df1 = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-2A', 'post_survey.csv'), sep=',')
            df1['expName'] = 'rn-iii-pilot-2A'
            df2 = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-2B', 'post_survey.csv'), sep=',')
            df2['expName'] = 'rn-iii-pilot-2B'
            post_survey_df = pd.concat([df2, df1], ignore_index=True)
            post_survey_df.to_csv(os.path.join(ddir, 'for_plots', 'rn-iii-pilot-2', 'post_survey_df.csv'), sep=',',
                                  index=False)
        elif self.experiment_name.startswith('rn-iii-pilot-3'):
            post_survey_df = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-3B', 'post_survey.csv'), sep=',')
            post_survey_df['expName'] = 'rn-iii-pilot-3B'
            post_survey_df.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'post_survey_df.csv'),
                                  sep=',',
                                  index=False)
        elif self.experiment_name.startswith('rn-iii-pilot-4'):
            post_survey_df = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-4b', 'post_survey.csv'), sep=',')
            post_survey_df['expName'] = 'rn-iii-pilot-4B'
            post_survey_df.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'post_survey_df.csv'),
                                  sep=',',
                                  index=False)
        elif self.experiment_name.startswith('rn-iii-pilot-5'):
            # edit experiment name with specific pilot version e.g. A-B-C
            post_survey_dfA = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5a', 'post_survey.csv'), sep=',')
            post_survey_dfA['expName'] = 'rn-iii-pilot-5A'
            post_survey_dfB = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5b', 'post_survey.csv'), sep=',')
            post_survey_dfB['expName'] = 'rn-iii-pilot-5B'
            post_survey_dfC = pd.read_csv(os.path.join(ddir, 'final', 'rn-iii-pilot-5c', 'post_survey.csv'), sep=',')
            post_survey_dfC['expName'] = 'rn-iii-pilot-5C'
            post_survey_df = pd.concat([post_survey_dfA, post_survey_dfB, post_survey_dfC], ignore_index=True)
            post_survey_df.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'post_survey_df.csv'),
                                  sep=',',
                                  index=False)

    def create_moves_scores_table(self):
        """
        Creates moves + scores table like the Google Sheets
        """
        # read in moves and scores

        scores_df = pd.read_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'scores.csv'), sep=',')
        moves = pd.read_csv(os.path.join(ddir, 'final', self.experiment_name, 'moves.csv'), sep=',', index_col=0)

        df = scores_df[scores_df['trialType'].isin(['individual', 'try_yourself', 'demonstration']) & scores_df[
            'isAI'] == False].reset_index(drop=True)

        # add columns move_1 to move_8
        for i in range(1, 9):
            df[f'move_{i}'] = np.nan

        # add moves from the moves dataframe to df based on the session id and trial index
        for i in range(1, 9):
            for idx, row in df.iterrows():
                try:
                    print(moves[
                              (moves['sessionId'] == row['sessionId']) &
                              (moves['trialID'] == row['trialIdx']) &
                              (moves['step'] == i)
                              ]['reward'].values)
                    df.loc[idx, f'move_{i}'] = moves[
                        (moves['sessionId'] == row['sessionId']) &
                        (moves['trialID'] == row['trialIdx']) &
                        (moves['step'] == i)
                        ]['reward'].values[0]
                except:
                    print(row)

        # add short session id with the last 3 characters
        df['playerId'] = df['sessionId'].str[-3:]

        # check that id is unique
        assert len(df['playerId'].unique()) == len(df['sessionId'].unique())

        # reorganize columns
        df = df[
            ['playerId', 'trialIdx', 'trialType', 'score', 'maxLevelReached', 'move_1', 'move_2', 'move_3', 'move_4',
             'move_5', 'move_6', 'move_7', 'move_8', 'networkId', 'myopic_score', 'ai_score', 'sessionId']]

        # sort by session id and trial index
        df = df.sort_values(by=['sessionId', 'trialIdx']).reset_index(drop=True)

        # save table
        df.to_csv(os.path.join(ddir, 'for_plots', self.experiment_name[:-1], 'scores_moves.csv'), sep=',', index=False)


#####################################
#####################################

if __name__ == '__main__':
    # directory management
    wdir = os.getcwd()
    ddir = os.path.join(wdir, 'data')
    fdir = os.path.join(wdir, 'analysis', 'figures')

    # read constants from environment variables + specify data path
    load_dotenv()
    EXPERIMENT_TYPE = os.environ['EXPERIMENT_TYPE']
    data_path = glob.glob(os.path.join(ddir, 'raw', EXPERIMENT_TYPE) + f'/sessions_{EXPERIMENT_TYPE}*.json')[0]

    # initialize Parser object
    P = Parser(data_path, EXPERIMENT_TYPE)
    P.open_data_file()
    #
    # # create base tables
    # P.create_base_table_player()
    # P.create_base_table_network()
    # P.create_base_table_trial()
    # P.create_base_table_moves()
    # P.create_written_strategy_table()
    # P.create_post_survey_table()
    # P.create_advisor_table()
    #
    # # P.examine_missing()
    # # P.create_aggregated_tables()
    #
    # P.create_data_for_plots()
    P.create_moves_scores_table()
