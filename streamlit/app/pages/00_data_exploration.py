import pandas as pd
import streamlit as st
import os
import yaml
import json
from pathlib import Path

from common.generate.generation import NetworkGenerator
from common.models.environment import Environment
from common.solve.rule_based import RuleAgent

from network_component.network_component import network_component
from plotting.plotting_solutions import plot_final_rewards, plot_avg_reward_per_step
from utils.io import load_yaml

st.set_page_config(page_title="RN III Task Explorer", layout="wide")

st.write(
    """
            # RN III Task Explorer
            This is an interactive application to explore stimuli and task
            design for the Reward Networks III project.
         """
)

# print current workdir
st.write(f"Current working directory: {os.getcwd()}")


if "moves_df" not in st.session_state:
    moves_df = pd.read_csv(os.path.join('analysis/data/experiment/processed/moves.csv'))

if "networks" not in st.session_state:
    networks_file = 'data/24_02_04/solution__valid_networks.json'
    with open(networks_file) as f:
        networks = json.load(f)
    networks = {n['network_id']: n for n in networks}
    st.session_state.networks = networks

if 'move_id' not in st.session_state:
    st.session_state.move_id = 0

if 'rerender_counter' not in st.session_state:
    st.session_state.rerender_counter = 0

# ------------------------------------------------------------------------------
#                      sidebar: generate and download options
# ------------------------------------------------------------------------------
with st.sidebar:
    st.write("## Select Condition")

    possible_conditions = moves_df['condition'].unique()
    
    condition = st.selectbox("Select condition", possible_conditions)
    
    sel_moves_df = moves_df[moves_df['condition'] == condition]
    
    st.write("## Select generation")
    
    possible_generations = sel_moves_df['generation'].unique()
    
    generation = st.selectbox("Select generation", possible_generations)
    
    sel_moves_df = sel_moves_df[sel_moves_df['generation'] == generation]
    
    st.write("## Select Session")
    
    possible_sessions = sel_moves_df['session_name'].unique()
    
    session = st.selectbox("Select session", possible_sessions)
    
    sel_moves_df = sel_moves_df[sel_moves_df['session_name'] == session]    
    
    st.write("## Select Trial Type")
    
    possible_trial_types = sel_moves_df['trial_type'].unique()
    
    trial_type = st.selectbox("Select trial type", possible_trial_types)
    
    sel_moves_df = sel_moves_df[sel_moves_df['trial_type'] == trial_type]
    
    st.write("Select trial id")
    
    possible_trial_id = sel_moves_df['trial_id'].unique()
    
    trial_id = st.selectbox("Select trial id", possible_trial_id)
    
    sel_moves_df = sel_moves_df[sel_moves_df['trial_id'] == trial_id]
    
    
    network_id = sel_moves_df['network_id'].values[0]
    source_num = sel_moves_df['source_num'].values
    target_num = sel_moves_df['target_num'].values
    reward = sel_moves_df['reward'].values
    cum_reward = sel_moves_df['reward'].cumsum().values
    moves = [int(m) for m in [source_num[0], *list(target_num)]]
    
    trial_name = f"{condition}_{generation}_{session}_{trial_type}_{trial_id}"

if "networks" in st.session_state:
    move_id = st.session_state.move_id

    with st.form("vizualization_form_wo_full", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            prev_net = st.form_submit_button("Show previous move")
        with col2:
            next_net = st.form_submit_button("Show next move")
        with col3:
            restart_clicked = st.form_submit_button("Restart")

        if restart_clicked:
            st.session_state.rerender_counter += 1
            

        if next_net:
            if st.session_state.move_id < 10:
                move_id += 1
                st.session_state.move_id = move_id
        if prev_net:
            if st.session_state.move_id > 0:
                move_id -= 1
                st.session_state.move_id = move_id

        network_component(
            type="default",
            network=st.session_state.networks[network_id],
            max_step=10,
            rerender_counter=st.session_state.rerender_counter,
            moves=moves[:move_id + 1],
            trial_name=trial_name
        )

