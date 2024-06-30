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
    st.write("## Select generation")

    st.write("## Select network")

    possible_networks = list(st.session_state.networks.keys())

    network_id = st.selectbox("Select network", possible_networks)

    agent_names = ["myopic", "take_loss"]

    agent_name = st.selectbox("Select network", agent_names)


    gen_params = {
        "n_steps": 10,
        "n_losses": 3,
        "rewards": [-50, 0, 100, 200, 400],
        "solution_columns": [
            "network_id",
            "strategy",
            "step",
            "source_node",
            "current_node",
            "reward",
            "total_reward",
        ]
    }

    networks = [st.session_state.networks[network_id]]

    agent = RuleAgent(networks, agent_name, gen_params)

    solutions = agent.solve()


    st.write("## Select move")

    st.session_state.move_id = st.slider("Select move", 0, 10, 0)


    print(solutions)
    st.write(solutions)
    st.write(networks[0]['starting_node'])


    source_node = solutions['source_node'].values
    current_node = solutions['current_node'].values
    moves = [int(m) for m in [source_node[0], *list(current_node)]]

    trial_name = f"{network_id}_{agent_name}"



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

