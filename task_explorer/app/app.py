import pandas as pd
import streamlit as st
import yaml

from generate.generation import NetworkGenerator
from models.environment import Environment
from network_component.network_component import network_component
from plotting.plotting_solutions import plot_final_rewards, \
    plot_avg_reward_per_step
from solve.rule_based import RuleAgent
from utils.io import load_yaml

st.set_page_config(page_title="RN III Task Explorer", layout="wide")

st.write("""
            # RN III Task Explorer
            This is an interactive application to explore stimuli and task 
            design for the Reward Networks III project. 
         """)

if "gen_env" not in st.session_state:
    environment = load_yaml("app/default_environment.yml")
    st.session_state.gen_env = Environment(**environment)

# ------------------------------------------------------------------------------
#                      sidebar: generate and download options
# ------------------------------------------------------------------------------
with st.sidebar:
    st.write("## Generate Networks")
    gen_params = {}
    data = None

    with st.expander("Upload environment file", expanded=False):
        with st.form(key="upload_params"):
            file = st.file_uploader("Upload environment parameters file",
                                    type="yml")
            submit_file = st.form_submit_button(label="Submit")

            if submit_file:
                if file is not None:
                    try:
                        data = yaml.safe_load(file)
                        st.success("File uploaded successfully")
                        st.session_state.gen_env = Environment(**data)
                    except Exception as e:
                        st.error(f"Error: {e}")

                else:
                    st.error("Please upload a file")

    # submit parameters for generation
    with st.form("generate_form", clear_on_submit=False):
        st.write("### Generate Parameters")
        st.write("Select the generation parameters")
        # how many networks to generate?
        gen_params['n_networks'] = st.number_input(
            label='How many networks do you want to generate?',
            min_value=1,
            max_value=100_000,
            value=11,
            step=10)

        gen_params['n_losses'] = st.number_input(
            label='How many large losses to take (for loss solving strategy)?',
            min_value=1,
            max_value=5,
            value=1,
            step=1)

        changed_env = st.session_state.gen_env.dict()
        changed_env['n_steps'] = st.number_input(
            label='How many step?',
            min_value=1,
            max_value=20,
            value=changed_env['n_steps'],
            step=1)

        for key, value in changed_env.items():
            if key == 'levels':
                with st.expander('Levels'):
                    for i, level in enumerate(value):
                        changed_env['levels'][i][
                            'min_n_nodes'] = st.number_input(
                            label=f'Min nodes in level {i}?',
                            min_value=1,
                            max_value=20,
                            value=int(level['min_n_nodes']),
                            step=1)

                        if level['max_n_nodes']:
                            changed_env['levels'][i][
                                'max_n_nodes'] = st.number_input(
                                label=f'Max nodes in level {i}',
                                min_value=1,
                                max_value=20,
                                value=int(level['max_n_nodes']),
                                step=1)

            if key == "rewards":
                with st.expander('Rewards'):
                    for f, _reward in enumerate(value):
                        # convert to string without brackets
                        reward = str(_reward['reward'])
                        reward = st.text_input(
                            label=f"Reward {f + 1}",
                            value=reward)
                        changed_env['rewards'][f]['reward'] = int(reward)

            if key == 'edges':
                with st.expander('Edges: level transition rewards'):
                    for f, from_level in enumerate(value):
                        # convert to string without brackets
                        rewards = str(from_level['rewards'])[1:-1]
                        lab = f"Rewards for transition from level" \
                              f" {from_level['from_level']} to level" \
                              f" {from_level['to_levels'][0]}:"
                        list_of_r = st.text_input(label=lab, value=rewards)
                        # convert to list of ints
                        list_of_rewards = [int(l) for l in list_of_r.split(',')]

                        changed_env['edges'][f]["rewards"] = list_of_rewards

                        # download title
        st.write("### Download Networks Options")

        # download the data yes or no?
        to_download_data = st.checkbox("Download the generated networks")
        # download the data yes or no?
        to_download_solutions = st.checkbox(
            "Download the generated networks' solutions")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Generate")

        if submitted:
            try:
                st.session_state.gen_env = Environment(**changed_env)
            except Exception as e:
                st.error(e)
                environment = load_yaml("app/default_environment.yml")
                st.session_state.gen_env = Environment(**environment)

            # Network_Generator class
            net_generator = NetworkGenerator(st.session_state.gen_env)
            networks = net_generator.generate(gen_params['n_networks'])
            networks = [n.dict() for n in networks]

            # check if the size of the networks is valid
            if len(networks) != gen_params['n_networks']:
                st.error(
                    f"The number of generated networks {len(networks)} is not "
                    f" equal to the number of networks requested "
                    f"{gen_params['n_networks']}")

            # update starting nodes
            for i in range(len(networks)):
                networks[i]['nodes'][networks[i]['starting_node']][
                    'starting_node'] = True
            print(f"N nets: {len(networks)}")
            st.session_state.networks = networks
            st.session_state.net_id = 1
            st.success("Networks generated!")
            if to_download_data:
                st.session_state.net_data = net_generator.save_as_json()

            gen_params['rewards'] = [r['reward'] for r in
                                     st.session_state.gen_env.dict()['rewards']]
            gen_params['n_steps'] = st.session_state.gen_env.n_steps

            # Solve networks with strategies
            myopic_agent = RuleAgent(networks, "myopic", gen_params)

            st.session_state.myopic_solutions = myopic_agent.solve()
            st.session_state.myopic_solutions_to_download = myopic_agent.save_solutions_frontend()
            loss_agent = RuleAgent(networks, "take_first_loss", gen_params)
            st.session_state.loss_solutions = loss_agent.solve()
            st.session_state.loss_solutions_to_download = loss_agent.save_solutions_frontend()
            st.success("Solutions to networks calculated!")

    # download button cannot be used inside form
    if to_download_data:
        st.download_button(
            label="Download data as JSON",
            data=st.session_state.net_data,
            file_name='networks.json')
    if to_download_solutions:
        st.download_button(
            label="Download solutions (myopic)",
            data=st.session_state.myopic_solutions_to_download,
            file_name='solutions_myopic.json')
        st.download_button(
            label="Download solutions (loss)",
            data=st.session_state.loss_solutions_to_download,
            file_name='solutions_loss.json')

    st.download_button(
        label="Download environment config",
        data=yaml.dump(st.session_state.gen_env.dict()),
        file_name="environment.yml",
    )

# ------------------------------------------------------------------------------
#                                   Compare
# ------------------------------------------------------------------------------
with st.expander("Compare strategies 🤖"):
    if "networks" in st.session_state:
        # create solution data file with all strategies in one file
        m_df = st.session_state.myopic_solutions
        l_df = st.session_state.loss_solutions
        strategy_data = pd.concat([m_df, l_df], ignore_index=True)
        strategy_data_final = strategy_data[strategy_data['step'] == st.session_state.gen_env.n_steps]
        st.write(f"Steps to solve: {st.session_state.gen_env.n_steps}")

        col1, col2 = st.columns([1, 2])
        g = plot_final_rewards(strategy_data_final)
        g3 = plot_avg_reward_per_step(strategy_data)
        with col1:
            st.pyplot(g)
        with col2:
            st.pyplot(g3)
    else:
        st.info("Please generate networks first!")

# ------------------------------------------------------------------------------
#                            Visualize Networks
# ------------------------------------------------------------------------------
with st.expander("Try yourself to solve the network 😎"):
    if "networks" in st.session_state:
        nets = st.session_state.networks
        net_id = st.session_state.net_id
        print(net_id)
        print(len(nets))

        with st.form("vizualization_form", clear_on_submit=False):
            col1, col2, _ = st.columns(3)
            with col1:
                prev_net = st.form_submit_button("Show previous network")
            with col2:
                next_net = st.form_submit_button("Show next network")

            if next_net:
                if st.session_state.net_id < len(nets):
                    net_id += 1
                    st.session_state.net_id = net_id
            if prev_net:
                if st.session_state.net_id > 0:
                    net_id -= 1
                    st.session_state.net_id = net_id

            network_component(
                timer=60,
                network=st.session_state.networks[st.session_state.net_id - 1],
                max_step=st.session_state.gen_env.n_steps,
                rewards=[r.reward for r in st.session_state.gen_env.rewards]
            )
    else:
        network_component(60)

with st.expander("Show solution dataframes 📊"):
    if "networks" in st.session_state:
        # ---metrics----
        st.markdown(
            "#### Average final reward obtained per strategy + "
            "average reward obtained at each step per strategy")
        col1, col2 = st.columns(2)
        with col1:
            avg_val1 = m_df[m_df['step'] == st.session_state.gen_env.n_steps]['total_reward'].mean().round(0)
            st.metric(
                "Myopic",
                value=int(avg_val1))
            m_avg_step_reward = m_df.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(axis=0)
            m_avg_step_reward.columns = ['Avg reward']
            st.dataframe(m_avg_step_reward)

        with col2:
            avg_val2 = l_df[l_df['step'] == st.session_state.gen_env.n_steps]['total_reward'].mean().round(0)
            st.metric(
                "Take Loss then Myopic",
                value=int(avg_val2))
            l_avg_step_reward = l_df.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(axis=0)
            l_avg_step_reward.columns = ['Avg reward']
            st.dataframe(l_avg_step_reward)
        #
        # with col3:
        #     st.metric("Random", "TODO")

        st.write("## Myopic solutions")
        st.dataframe(st.session_state.myopic_solutions)
        st.write("## Take first loss solutions")
        st.dataframe(st.session_state.loss_solutions)
    else:
        st.info("Please generate networks first!")
