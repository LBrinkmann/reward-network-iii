import datetime
from typing import Optional, List, Union, Literal

from beanie import Document, PydanticObjectId
from pydantic import BaseModel,Field as PydanticField

from models.trial import Trial


class Session(BaseModel):
    id: PydanticObjectId = PydanticField(default_factory=PydanticObjectId, alias="_id")
    created_at: datetime.datetime = datetime.datetime.now()
    experiment_num: int
    experiment_type: str = 'reward_network_iii'
    # id of the experiment settings (config) used for this session
    config_id: Optional[PydanticObjectId]
    generation: int
    session_num_in_generation: int
    ai_player: Optional[bool] = False
    subject_id: Optional[PydanticObjectId]
    average_score: Optional[int]
    trials: List[Trial]
    current_trial_num: Optional[int] = 0
    advise_ids: Optional[Union[List[PydanticObjectId]]] = []
    child_ids: Optional[Union[List[PydanticObjectId]]] = []
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
        schema_extra = {
            "example": {}
        }
#
# import datetime
# from typing import Optional, Union, Literal
#
# from beanie import Document,PydanticObjectId
# from pydantic import BaseModel,Field as PydanticField
#
# from models.trial import Trial
#
#
# class Session(BaseModel):
#     id: PydanticObjectId = PydanticField(default_factory=PydanticObjectId, alias="_id")
#     created_at: datetime.datetime = datetime.datetime.now()
#     experiment_num: int
#     experiment_type: str = 'reward_network_iii'
#     generation: int
#     session_num_in_generation: int
#     ai_player: Optional[bool] = False
#     subject_id: Optional[PydanticObjectId]
#     average_score: Optional[int]
#     trials: list[Trial]
#     current_trial_num: Optional[int] = 0
#     advise_ids: Optional[list[PydanticObjectId]] = []
#     child_ids: Optional[list[PydanticObjectId]] = []
#     unfinished_parents: Optional[int] = 0
#     finished: Optional[bool] = False
#     finished_at: Optional[datetime.datetime]
#     available: Optional[bool] = False  # available for subject to play
#     started_at: Optional[datetime.datetime]  # when the first trial was started
#     expired: Optional[bool] = False  # if the session is expired
#     replaced: Optional[bool] = False  # if the session was replaced
#     # time spent on the session after the session was finished
#     time_spent: Optional[datetime.timedelta] = datetime.timedelta(0)
#
#     class Config:
#         # TODO: add example
#         schema_extra = {
#             "example": {}
#         }


class SessionError(BaseModel):
    message: Literal[
        'No available session for the subject',
        'Multiple subjects with the same prolific id'
    ]
