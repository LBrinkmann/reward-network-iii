from datetime import datetime
from typing import List

from beanie import PydanticObjectId

from models.session import Session
from models.trial import (
    Trial,
    Solution,
    TrialError,
    WrittenStrategy,
    Advisor,
    PostSurvey,
)
from utils.utils import estimate_solution_score

MAX_STEPS = 10


async def save_trial(body, session, trial, trial_type):
    # save trial results
    if trial_type == "individual":
        save_individual_demonstration_trial(trial, body)
    elif trial_type == "social_learning_selection":
        session = await save_social_learning_selection(body, session, trial)
    elif trial_type == "observation":
        save_individual_demonstration_trial(trial, body)
    elif trial_type == "repeat":
        save_individual_demonstration_trial(trial, body)
    elif trial_type == "try_yourself":
        save_individual_demonstration_trial(trial, body)
    elif trial_type == "demonstration":
        save_individual_demonstration_trial(trial, body)
    elif trial_type == "written_strategy":
        save_written_strategy(trial, body)
    elif trial_type == "post_survey":
        save_survey_trial(trial, body)
    elif trial_type in ["consent", "practice", "debriefing", "instruction"]:
        save_empty_trial(trial)

    if trial.solution is not None:
        score = estimate_solution_score(trial.network, trial.solution.moves, MAX_STEPS)
        assert score > -100_000, "invalid move sequence"
    # update session with the trial
    session.trials[session.current_trial_num] = trial


def save_individual_demonstration_trial(trial: Trial, body: Solution):
    if not isinstance(body, Solution):
        return TrialError(message="Trial results are missing")

    trial.solution = Solution(
        moves=body.moves,
        correctRepeats=body.correctRepeats,
        score=estimate_solution_score(trial.network, body.moves, MAX_STEPS),
        trial_id=trial.id,
        finished_at=datetime.now(),
    )
    trial.finished_at = datetime.now()
    trial.finished = True


def save_written_strategy(trial: Trial, body: WrittenStrategy):
    if not isinstance(body, WrittenStrategy):
        return TrialError(message="Trial results are missing")

    trial.written_strategy = WrittenStrategy(
        strategy=body.strategy, trial_id=trial.id, finished_at=datetime.now()
    )
    trial.finished_at = datetime.now()
    trial.finished = True


def save_survey_trial(trial: Trial, body: PostSurvey):
    if not isinstance(body, PostSurvey):
        return TrialError(message="Trial results are missing")

    trial.post_survey = PostSurvey(
        questions=body.questions, trial_id=trial.id, finished_at=datetime.now()
    )
    trial.finished_at = datetime.now()
    trial.finished = True


async def save_social_learning_selection(body: Advisor, session: Session, trial: Trial):
    if not isinstance(body, Advisor):
        return TrialError(message="Trial results are missing")

    social_learning_block_idx = trial.social_learning_block_idx

    social_learning_trails = [
        t
        for t in session.trials
        if t.trial_type in ["try_yourself", "observation", "repeat"]
        and t.social_learning_block_idx == social_learning_block_idx
    ]

    # get max block_network_idx
    sl_idx_max = max([t.block_network_idx for t in social_learning_trails])

    # get advisor session
    ad_s = await Session.get(body.advisor_id)

    if ad_s is None:
        return TrialError(message="Advisor session is not found")

    # sort by trail_id
    ad_trials = sorted(ad_s.trials, key=lambda t: t.id)

    # get advisor demonstration trials
    ad_demo_trials = [t for t in ad_trials if t.trial_type == "demonstration"]

    assert len(ad_demo_trials) >= sl_idx_max, f"{len(ad_demo_trials)} <= {sl_idx_max}"

    # select advisor's demonstration trial
    ad_demo_trials = ad_demo_trials[-sl_idx_max - 1 :]

    assert (
        len(ad_demo_trials) == sl_idx_max + 1
    ), f"{len(ad_demo_trials)} == {sl_idx_max + 1}"

    ad_written_strategies = [
        t.written_strategy for t in ad_trials if t.trial_type == "written_strategy"
    ]

    # select advisor's written strategy
    ad_written_strategy = ad_written_strategies[-1]

    for i, ad_demo_trail in enumerate(ad_demo_trials):
        # update `selected_by_children` field for advisor's demonstration trial
        ad_demo_trail.selected_by_children.append(session.id)

        advisor = Advisor(
            advisor_id=body.advisor_id,
            solution=ad_demo_trail.solution,
        )
        for sl_trial in social_learning_trails:
            if sl_trial.block_network_idx != i:
                continue
            sl_trial.advisor = advisor
            sl_trial.network = ad_demo_trail.network
            assert session.trials[sl_trial.id].id == sl_trial.id
            session.trials[sl_trial.id] = sl_trial
            assert session.trials[sl_trial.id].network is not None

    trial.advisor = Advisor(
        advisor_id=body.advisor_id, written_strategy=ad_written_strategy.strategy
    )

    trial.finished_at = datetime.now()
    trial.finished = True
    session.trials[trial.id] = trial
    for t in session.trials:
        if t.trial_type in ["individual", "try_yourself", "observation"]:
            assert (
                t.network is not None
            ), f"{t.id} {t.trial_type} {t.social_learning_block_idx} {sl_idx_max}"
    return session


def save_empty_trial(trial: Trial):
    trial.finished_at = datetime.now()
    trial.finished = True
