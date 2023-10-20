import json

import streamlit.components.v1 as components

BASE_URL = "https://631063bad8ec25de99f6946b-wordcsbsge.chromatic.com/" \
           "iframe.html?"
STORY_COMPONENT = "id=utils-taskexplorer2--default&viewMode=story"


def network_component(timer: int = 25, network: dict = None, max_step: int = 8,
                      rewards: list = [-100, -20, 0, 20, 140]):
    """Embeds a network component from Chromatic.

    Parameters
    ----------
    timer : int
        The time in ms for the component to wait before rendering.
    network : dict
        The network to be rendered.
    max_step : int
        The maximum number of steps in one trial.
    rewards : list
    """

    url = f"{BASE_URL}args=timer:{timer};maxSteps:{max_step}" \
          f"&{STORY_COMPONENT}"
    if network is not None:
        # convert dict to string
        # set separators=(',', ':') to remove spaces
        network_args = json.dumps(network, separators=(',', ':'))
        url += f"&custom_args={network_args}"

    rewards_args = json.dumps(rewards, separators=(',', ':'))
    url += f"&custom_rewards={rewards_args}"

    components.iframe(url, height=700, width=800)
