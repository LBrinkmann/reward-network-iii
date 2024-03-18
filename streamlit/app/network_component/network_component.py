import math
import os
from typing import List

import streamlit.components.v1 as components


def round_to_significant_figures(num, sig_figs):
    """Round to specified number of sigfigs."""
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # handle zero separately


def round_floats(o):
    if isinstance(o, float):
        return round_to_significant_figures(o, 2)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def network_component(
    type: str = "default",
    network: dict = None,
    max_step: int = 8,
    rerender_counter: int = 0,
    moves: List[int] = None,
    trial_name: str = None
):
    """Embeds a network component from Chromatic.

    Parameters
    ----------
    network : dict
        The network to be rendered.
    max_step : int
        The maximum number of steps in one trial.
    """

    BASE_URL = os.getenv("FRONTEND_URL", "http://localhost:9000")
    url = f"{BASE_URL}/streamlit"
    network_component = components.declare_component(
        "network_component",
        url=url,
    )
    network_component(
        network=round_floats(network),
        maxMoves=max_step,
        showAllEdges=type == "legacy",
        rerenderCounter=rerender_counter,
        moves=moves,
        trial_name=trial_name
    )
