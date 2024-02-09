import datetime
from typing import Optional, List, Union, Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel

from models.trial import Trial


class Session(Document):
    created_at: datetime.datetime = datetime.datetime.now()
    experiment_num: int
    experiment_type: str = "reward_network_iii"
    priority: Optional[float] = 0
    condition: Optional[str] = None
    # id of the experiment settings (config) used for this session
    config_id: Optional[PydanticObjectId] = None
    generation: int
    session_num_in_generation: int
    ai_player: Optional[bool] = False
    simulated_subject: Optional[bool] = False
    subject_id: Optional[PydanticObjectId] = None
    average_score: Optional[int] = None
    trials: List[Trial]
    current_trial_num: Optional[int] = 0
    advise_ids: List[PydanticObjectId] = []
    child_ids: List[PydanticObjectId] = []
    finished: Optional[bool] = False
    completed: Optional[bool] = False
    finished_at: Optional[datetime.datetime] = None
    available: Optional[bool] = False  # available for subject to play
    started_at: Optional[datetime.datetime] = None # when the first trial was started
    expired: Optional[bool] = False  # if the session is expired
    replaced: Optional[bool] = False  # if the session was replaced
    is_replacement_for: Optional[PydanticObjectId] = None  # id of the session that was replaced
    # time spent on the session after the session was finished
    time_spent: Optional[datetime.timedelta] = datetime.timedelta(0)

    class Config:
        # TODO: add example
        schema_extra = {"example": {}}

    class Settings:
        indexes = [
            "subject_id",
            "experiment_type",
            ["available", "experiment_type"],
            # ["_id", "unfinished_parents"],
            ["finished", "subject_id", "expired"],
            "started_at",
            ["finished", "replaced", "time_spent"],
            ["expired", "replaced", "experiment_type"],
        ]
