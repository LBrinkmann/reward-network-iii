from datetime import datetime, timedelta
from typing import Union

from beanie.odm.operators.find.comparison import In
from beanie.odm.operators.update.general import Set

from models.config import ExperimentSettings
from models.session import Session
from models.trial import SessionError
from models.subject import Subject
from study_setup.generate_sessions import create_trials
from utils.utils import estimate_average_player_score


async def get_session(prolific_id) -> Union[Session, SessionError]:
    """ Get session for the subject """
    # check if collection Subject exists
    if await Subject.find().count() > 0:
        subjects_with_id = await Subject.find(
            Subject.prolific_id == prolific_id).to_list()
    else:
        subjects_with_id = []

    if len(subjects_with_id) == 0:
        # subject does not exist
        # creat a new subject
        subject = Subject(prolific_id=prolific_id)
        # save subject to database
        await subject.save()
        # session initialization for the subject
        # session will not be assigned to the subject if there is no available
        await initialize_session(subject)
    elif len(subjects_with_id) > 1:
        # if more than one subject with the same prolific id return error
        return SessionError(
            message=f'Prolific ID {prolific_id} already exists')
    else:
        subject = subjects_with_id[0]

    # get session for the subject
    session = await Session.find_one(Session.subject_id == subject.id)
    if session is None:
        # this happens when all available sessions are taken
        return SessionError(message='No available sessions')

    # this will happen only for a new subject
    if subject.session_id is None:
        # update Subject.session_id field if it is empty
        await subject.update(Set({Subject.session_id: session.id}))

    return session


async def initialize_session(subject: Subject):
    # find an active configuration
    config = await ExperimentSettings.find_one(
        ExperimentSettings.active == True)
    # Check and remove expired sessions
    await replace_stale_session(config)

    # assign subject to any available session
    await Session.find_one(
        Session.available == True,
        # select session for this experiment
        Session.experiment_type == config.experiment_type
    ).update(
        Set({
            Session.available: False,
            Session.subject_id: subject.id,
            Session.current_trial_num: 0,
            Session.started_at: datetime.now()  # save session start time
        })
    )


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


async def end_session(session):
    session.finished_at = datetime.now()
    session.finished = True
    session.average_score = estimate_average_player_score(session)
    session.time_spent = session.finished_at - session.started_at
    # save session
    await session.save()

    # Ensure that the expired but finished sessions are removed from the
    # session tree
    config = await ExperimentSettings.find_one(
        ExperimentSettings.active == True)
    # Check and remove expired sessions
    await replace_stale_session(config)

    # get the updated version of the session
    session = await Session.get(session.id)
    # if session in not finished then it was expired and replaced
    # no need to update availability status of child sessions
    if not session.finished:
        return
    # update child sessions
    await update_availability_status_child_sessions(session)


async def update_availability_status_child_sessions(session: Session):
    """ Update child sessions availability status """

    # update `unfinished_parents` value for child sessions
    await Session.find(
        In(Session.id, session.child_ids)
    ).inc({Session.unfinished_parents: -1})

    # update child sessions status if all parent sessions are finished
    await Session.find(
        In(Session.id, session.child_ids),
        Session.unfinished_parents == 0
    ).update(Set({Session.available: True}))


async def replace_stale_session(exp_config: ExperimentSettings,
                                time_delta: int = 30):
    """
    Replace the unfinished session so as not to break the social learning chains

    Parameters
    ----------
    exp_config: ExperimentSettings
        Experiment settings
    time_delta : int
        Time in minutes after which the session is considered expired
    """
    # get all expired sessions (sessions that were started long ago)
    await Session.find(
        Session.finished == False,  # session is not finished
        Session.subject_id != None,  # session is assigned to subject
        # there can be old expired and already replaces sessions
        Session.expired == False
    ).find(
        # find all sessions older than the specified time delta
        Session.started_at < datetime.now() - timedelta(minutes=time_delta)
    ).update(Set({Session.expired: True}))

    # mark as expired finished but not replaced sessions
    await Session.find(
        Session.finished == True,  # session is finished
        Session.replaced == False  # session is not replaced
    ).find(
        Session.time_spent > timedelta(minutes=time_delta)
    ).update(Set({Session.expired: True}))

    # get all newly expired sessions
    new_expired = await Session.find(
        # session is marked as expired
        Session.expired == True,
        # session has not yet been replaced
        Session.replaced == False
    ).to_list()

    # nothing to replace
    if len(new_expired) == 0:
        return

    # make empty duplicates of the expired sessions
    for s in new_expired:
        # create an empty session to replace the expired one
        new_s = create_trials(
            config_id=exp_config.id,
            experiment_num=s.experiment_num,
            experiment_type=s.experiment_type,
            generation=s.generation,
            session_idx=s.session_num_in_generation,
            n_social_learning_trials=exp_config.n_social_learning_trials,
            n_individual_trials=exp_config.n_individual_trials,
            n_demonstration_trials=exp_config.n_demonstration_trials,
            redirect_url=exp_config.redirect_url
        )
        new_s.advise_ids = s.advise_ids
        new_s.child_ids = s.child_ids
        new_s.unfinished_parents = 0
        new_s.available = True
        new_s.id = s.id  # copy id
        await new_s.replace()

        # mark the session as expired and delete id
        s.expired = True
        s.replaced = True
        del s.id
        await s.save()
