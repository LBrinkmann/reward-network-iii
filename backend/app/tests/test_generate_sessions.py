import httpx
import pytest

from models.config import ExperimentSettings
from models.session import Session
from study_setup.generate_sessions import (
    generate_sessions,
    create_trials,
    reset_networks,
)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.parametrize(
    "n_generations,n_ai_players,"
    "n_sessions_per_generation,n_advise_per_session,conditions, simulate_humans",
    [
        (1, 0, 10, 0, ["wo_ai"], False),  # pilot 6A (gen 0)
        (2, 3, 10, 5, ["w_ai"], True),  # pilot 6B (gen 1, w_ai)
        (2, 0, 10, 5, ["wo_ai"], True),  # pilot 6C (gen 1, wo_ai)
        (3, 3, 20, 5, ["wo_ai", "w_ai"], False),  # full experiment
    ],
)
async def test_generate_sessions(
    default_client: httpx.AsyncClient,
    e_config: ExperimentSettings,
    n_generations,
    n_ai_players,
    n_sessions_per_generation,
    n_advise_per_session,
    conditions,
    simulate_humans,
    experiment_type="reward_network_iii",
):
    # Clean up resources
    await Session.find().delete()
    sessions = await Session.find().first_or_none()
    assert sessions is None

    reset_networks()
    e_config.n_generations = n_generations
    e_config.n_ai_players = n_ai_players
    e_config.n_sessions_per_generation = n_sessions_per_generation
    e_config.n_advise_per_session = n_advise_per_session
    e_config.experiment_type = experiment_type
    e_config.conditions = conditions
    e_config.simulate_humans = simulate_humans

    await generate_sessions(
        experiment_num=0,
        config=e_config,
    )
    sessions = await Session.find().to_list()

    assert sessions is not None

    net_ids = []
    for s in sessions:
        assert s.experiment_type == "reward_network_iii"
        if s.generation != 0:
            # check the number of parents
            assert len(s.advise_ids) == n_advise_per_session
        # collect all network ids
        net_ids += [
            t.network.network_id
            for t in s.trials
            if t.network is not None and t.trial_type == "demonstration"
        ]

    # check that each network is unique
    assert len(net_ids) == len(set(net_ids))

    sessions = await Session.find({"generation": 0, "condition": "w_ai"}).to_list()
    assert len(sessions) == n_ai_players

    sessions = await Session.find({"generation": 0}).to_list()
    if len(e_config.conditions) == 1:
        assert len(sessions) == n_sessions_per_generation
    elif len(e_config.conditions) == 2:
        assert len(sessions) == (n_sessions_per_generation / 2 + n_ai_players)

    if simulate_humans:
        sessions = await Session.find({"generation": 0, "available": True}).to_list()
        assert len(sessions) == 0
        sessions = await Session.find({"generation": 0, "finished": False}).to_list()
        assert len(sessions) == 0

    for i in range(1, n_generations):
        sessions = await Session.find({"generation": i}).to_list()
        assert len(sessions) == n_sessions_per_generation

        if len(e_config.conditions) == 2:
            sessions = await Session.find(
                {"generation": i, "condition": "w_ai"}
            ).to_list()
            assert len(sessions) == n_sessions_per_generation / 2

            sessions = await Session.find(
                {"generation": i, "condition": "wo_ai"}
            ).to_list()
            assert len(sessions) == n_sessions_per_generation / 2
        else:
            sessions = await Session.find(
                {"generation": i, "condition": e_config.conditions[0]}
            ).to_list()
            assert len(sessions) == n_sessions_per_generation


@pytest.mark.asyncio
async def test_create_trials(
    default_client: httpx.AsyncClient, e_config: ExperimentSettings
):
    reset_networks()

    session = create_trials(
        experiment_num=0,
        session_idx=0,
        condition="wo_ai",
        generation=0,
        config=e_config,
    )

    for t in session.trials:
        assert t.trial_type not in [
            "social_learning_selection",
            "observation",
            "repeat",
            "try_yourself",
        ]
        assert t.trial_type in [
            "consent",
            "instruction",
            "demonstration",
            "written_strategy",
            "debriefing",
            "individual",
            "post_survey",
            "practice",
        ]

    session = create_trials(
        experiment_num=0,
        session_idx=0,
        condition="wo_ai",
        generation=1,
        config=e_config,
    )

    for t in session.trials:
        assert t.trial_type in [
            "consent",
            "instruction",
            "demonstration",
            "written_strategy",
            "debriefing",
            "individual",
            "post_survey",
            "practice",
            "social_learning_selection",
            "observation",
            "repeat",
            "try_yourself",
        ]
