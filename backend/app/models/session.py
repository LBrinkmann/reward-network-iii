import datetime
from typing import Optional, List, Union, Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel

from models.trial import Trial


class Session(Document):
    created_at: datetime.datetime = datetime.datetime.now()
    experiment_num: int
    experiment_type: str = "reward_network_iii"
    # id of the experiment settings (config) used for this session
    config_id: Optional[PydanticObjectId]
    generation: int
    session_num_in_generation: int
    ai_player: Optional[bool] = False
    subject_id: Optional[PydanticObjectId]
    average_score: Optional[int]
    trials: List[Trial]
    current_trial_num: Optional[int] = 0
    advise_ids: List[PydanticObjectId] = []
    child_ids: List[PydanticObjectId] = []
    unfinished_parents: Optional[int] = 0
    finished: Optional[bool] = False
    finished_at: Optional[datetime.datetime]
    available: Optional[bool] = False  # available for subject to play
    started_at: Optional[datetime.datetime]  # when the first trial was started
    expired: Optional[bool] = False  # if the session is expired
    replaced: Optional[bool] = False  # if the session was replaced
    # time spent on the session after the session was finished
    time_spent: Optional[datetime.timedelta] = datetime.timedelta(0)

    class Config:
        # TODO: add example
        schema_extra = {"example": {}}
