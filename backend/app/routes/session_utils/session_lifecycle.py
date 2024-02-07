from datetime import datetime, timedelta
from typing import Union

from beanie import PydanticObjectId
from beanie.odm.operators.find.comparison import In
from beanie.odm.operators.find.array import Size
from beanie.odm.operators.update.general import Set
from beanie.odm.operators.update.array import AddToSet
from beanie.odm.queries.update import UpdateResponse

from models.config import ExperimentSettings
from models.session import Session
from models.trial import SessionError
from models.subject import Subject
from study_setup.generate_sessions import create_trials
from utils.utils import estimate_average_player_score
from models.trial import Solution, Trial

# This is hard coded, but it should be a parameter of the experiment
MAX_STEPS = 10

async def get_session(prolific_id, experiment_type=None) -> Union[Session, SessionError]:
    """Get session for the subject"""
    # check if collection Subject exists
    if await Subject.find().count() > 0:
        subjects_with_id = await Subject.find(
            Subject.prolific_id == prolific_id
        ).to_list()
    else:
        subjects_with_id = []

    if len(subjects_with_id) == 0:
        assert experiment_type is not None, "Experiment type is not specified"
        # subject does not exist
        # creat a new subject
        subject = Subject(prolific_id=prolific_id)
        # save subject to database
        await subject.save()
        # session initialization for the subject
        # session will not be assigned to the subject if there is no available
        await initialize_session(subject, experiment_type=experiment_type)
    elif len(subjects_with_id) > 1:
        # if more than one subject with the same prolific id return error
        return SessionError(message=f"Prolific ID {prolific_id} already exists")
    else:
        subject = subjects_with_id[0]

    # get session for the subject
    try:
        session = await Session.find_one(Session.subject_id == subject.id)
    except Exception as e:
        print(f"Error when loading session {subject.id}", flush=True)
        raise e
    if session is None:
        print("Find session for the subject", flush=True)
        # try to initialize session for the subject
        await initialize_session(subject, experiment_type=experiment_type)
        # get session for the subject
        session = await Session.find_one(Session.subject_id == subject.id)
        if session is None:
            print("No available sessions", flush=True)
            # this happens when all available sessions are taken
            return SessionError(message="No available sessions")

    # this will happen only for a new subject
    if subject.session_id is None:
        # update Subject.session_id field if it is empty
        await subject.update(Set({Subject.session_id: session.id}))

    return session


async def initialize_session(subject: Subject, experiment_type: str):
    # find an active configuration
    config = await ExperimentSettings.find_one(ExperimentSettings.experiment_type == experiment_type)

    assert config is not None, "Experiment settings are not found"

    # Check and replace expired sessions
    await expire_stale_session(config)
    await replace_expired_sessions(config)

    # # # assign subject to any available session
    session = await Session.find_one(
        Session.available == True,
        # select session for this experiment
        Session.experiment_type == config.experiment_type,
    ).update(
        Set(
            {
                Session.available: False,
                Session.subject_id: subject.id,
                Session.current_trial_num: 0,
                Session.started_at: datetime.now(),  # save session start time
            }
        ),
        response_type=UpdateResponse.NEW_DOCUMENT,
        sort = [("priority", -1)]
    )

    # session = await Session.find_one(Session.subject_id == subject.id)
    if session is None:
        # no available sessions
        return
    print(f"Session assigned to the subject, {session.priority}", flush=True)
    # print(f"Other session, {sessions[0].priority}", flush=True)




async def update_session(session):
    # if this is the last trial minus debriefing trial
    if (session.current_trial_num + 1) == (len(session.trials) - 1):
        await end_session(session)
        # increase trial index by 1 to show debriefing trial
        session.current_trial_num += 1
        # save session
        await session.save()
    elif (session.current_trial_num + 1) == len(session.trials):
        pass
    else:
        # increase trial index by 1
        session.current_trial_num += 1

        # save session
        await session.save()



def check_moves_complete(solution: Solution) -> bool:
    """Check if the moves are complete"""
    return len(solution.moves) == (MAX_STEPS + 1) # +1 for the initial state


def check_solution_complete(trial: Trial) -> bool:
    """Check if the solution is complete"""
    return trial.solution is not None and check_moves_complete(trial.solution)


def check_all_demonstration_trials_complete(session: Session) -> bool:
    """Check if all demonstration trials are complete"""
    return all(
        check_solution_complete(trial)
        for trial in session.trials
        if trial.trial_type == "demonstration"
    )


async def end_session(session):
    config = await ExperimentSettings.find_one(ExperimentSettings.experiment_type == session.experiment_type)
    
    session.finished_at = datetime.now()
    session.time_spent = session.finished_at - session.started_at
    session.finished = True
    
    session.expired = session.time_spent > timedelta(minutes=config.session_timeout)
    
    if not session.expired:
        session.completed = check_all_demonstration_trials_complete(session)
        # if session is not completed then it needs to be replaced (so we set it as expired)
        session.expired = not session.completed
        session.average_score = estimate_average_player_score(session)

    # save session
    await session.save()

    # # Replace expired sessions
    # await replace_expired_sessions(config)

    # Only update child sessions if the session is completed
    if session.completed:
        # update child sessions
        await update_availability_status_child_sessions(session, config)


async def update_availability_status_child_sessions(session: Session, exp_config: ExperimentSettings):
    """Update child sessions availability status"""

    # update `unfinished_parents` value for child sessions
    await Session.find(In(Session.id, session.child_ids)).inc(
        {Session.unfinished_parents: -1}
    )

    # add finished parent to the list of finished parents
    await Session.find(In(Session.id, session.child_ids)).update(
        AddToSet({Session.finished_parents: session.id})
    )

    # update child sessions status if all parent sessions are finished
    await Session.find(
        In(Session.id, session.child_ids), Size(Session.finished_parents, exp_config.n_advise_per_session)
    ).update(Set({Session.available: True}))


async def expire_stale_session(exp_config: ExperimentSettings):
    """
    Replace the unfinished session so as not to break the social learning chains

    Parameters
    ----------
    exp_config: ExperimentSettings
        Experiment settings
    """
    time_delta = exp_config.session_timeout

    # get all expired sessions (sessions that were started long ago)
    res = await Session.find(
        Session.finished == False,  # session is not finished
        Session.subject_id != None,  # session is assigned to subject
        # there can be old expired and already replaces sessions
        Session.expired == False,
    # ).find(
        # find all sessions older than the specified time delta
        Session.started_at
        < datetime.now() - timedelta(minutes=time_delta)
    ).update(
        Set({Session.expired: True})
    )
    # print(res, flush=True)

    # mark as expired finished but not replaced sessions
    await Session.find(
        Session.finished == True,  # session is finished
        Session.replaced == False,  # session is not replaced
    ).find(Session.time_spent > timedelta(minutes=time_delta)).update(
        Set({Session.expired: True})
    )


async def replace_expired_sessions(exp_config: ExperimentSettings):
    """
    Replace expired sessions with new ones
    """
    
    while True:
        # get all newly expired sessions
        expired_session = await Session.find_one(
            # session is marked as expired
            Session.expired == True,
            # session has not yet been replaced
            Session.replaced == False,
            Session.experiment_type == exp_config.experiment_type,
        ).update(
            Set({Session.replaced: True}),
            response_type=UpdateResponse.NEW_DOCUMENT
        )

        if expired_session is None:
            break

        assert expired_session.expired == True, "Session is not expired"
        assert expired_session.replaced == True, "Session is not marked as replaced"


        old_session_copy = expired_session.copy()
        del old_session_copy.id
        await old_session_copy.insert()

        print(f"Session {expired_session.id} is expired", flush=True)
        print(f'Saved copy of the session {old_session_copy.id}', flush=True)

        # make empty duplicates of the expired sessions
        # create an empty session to replace the expired one
        new_s = create_trials(
            experiment_num=expired_session.experiment_num,
            generation=expired_session.generation,
            session_idx=expired_session.session_num_in_generation,
            condition=expired_session.condition,
            config=exp_config,
        )
        new_s.advise_ids = expired_session.advise_ids
        new_s.child_ids = expired_session.child_ids
        new_s.unfinished_parents = 0
        new_s.available = True
        new_s.is_replacement = True
        new_s.id = expired_session.id  # copy id

        # save new session
        await new_s.replace()
        print(f"Session {new_s.id} is created", flush=True)

