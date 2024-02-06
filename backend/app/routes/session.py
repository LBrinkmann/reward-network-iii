from typing import Union

from fastapi import APIRouter

from models.trial import Trial, Solution, TrialSaved, TrialError, \
    WrittenStrategy, Advisor, PostSurvey, SessionError
from .session_utils.prepare_trial import prepare_trial
from .session_utils.save_trial import save_trial
from .session_utils.session_lifecycle import update_session
from .session_utils.session_lifecycle import get_session

session_router = APIRouter(tags=["Session"])


@session_router.get('/{experiment_type}/{prolific_id}', response_model_by_alias=False)
async def get_current_trial(experiment_type: str, prolific_id: str) -> Union[Trial, SessionError]:
    """
    Get current trial from the session.
    """
    # find session and trial for the subject
    session = await get_session(prolific_id, experiment_type)

    # return error if session is not available
    if isinstance(session, SessionError):
        return session

    trial = await prepare_trial(session)

    try:
        await session.save()
    except Exception as e:
        print('save session', session, flush=True)
    return trial

@session_router.post('/{prolific_id}/{trial_id}')
async def post_current_trial_results(
        prolific_id: str,
        trial_id: int,
        body: Union[Solution, WrittenStrategy, Advisor, PostSurvey, None
        ] = None) -> Union[TrialSaved, SessionError, TrialError]:
    # find session assigned to the subject
    session = await get_session(prolific_id)

    # return error if session is not available
    if isinstance(session, SessionError):
        return session

    # get current trial
    trial = session.trials[session.current_trial_num]

    # check if trial type is correct
    if trial.id != trial_id:
        return TrialError(message='Trial type is not correct')

    await save_trial(body, session, trial, trial.trial_type)

    await update_session(session)

    return TrialSaved()
