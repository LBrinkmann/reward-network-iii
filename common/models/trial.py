import datetime
from typing import Optional, List, Dict

# https://github.com/pydantic/pydantic/issues/545
from typing_extensions import Literal

from beanie import PydanticObjectId

from common.models.network import Network
from pydantic import BaseModel


class Solution(BaseModel):
    moves: List[int]
    correctRepeats: Optional[List[bool]] = None
    score: Optional[int] = None # solution score
    trial_id: Optional[int] = None # trial number in session
    finished_at: Optional[datetime.datetime] = None
    solution_type: Optional[Literal["myopic", "loss", "machine_0", "machine_1", "machine_2"]] = None


class Advisor(BaseModel):
    advisor_id: PydanticObjectId  # advisor id
    solution: Optional[Solution] = None
    written_strategy: Optional[str] = None


class AdvisorSelection(BaseModel):
    advisor_ids: List[PydanticObjectId]  # advisor ids
    scores: List[int]  # scores for each advisor


class WrittenStrategy(BaseModel):
    strategy: str
    trial_id: Optional[int] = None # trial number in session
    finished_at: Optional[datetime.datetime] = None


class PostSurvey(BaseModel):
    questions: Dict[str, str]
    trial_id: Optional[int] = None  # trial number in session
    finished_at: Optional[datetime.datetime] = None


class Trial(BaseModel):
    id: int  # trial number in session
    trial_type: Literal[
        "consent",
        "instruction",
        "practice",
        "social_learning_selection",
        "observation",
        "repeat",
        "try_yourself",
        "individual",
        "demonstration",
        "written_strategy",
        "post_survey",
        "debriefing",
    ]
    # instruction trial relevant field
    instruction_type: Optional[
        Literal[
            "welcome",
            "learning_selection",
            "pre_social_learning",
            "individual",
            "practice_rounds",
            "pre_social_learning_gen0",
            "demonstration",
            "written_strategy",
            "written_strategy_start",
        ]
    ] = None
    finished: Optional[bool] = False
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    network: Optional[Network] = None
    solution: Optional[Solution] = None
    # social learning trial related field
    advisor: Optional[Advisor] = None
    # social learning selection trial relevant field
    advisor_selection: Optional[AdvisorSelection] = None
    # demonstration trial relevant field
    selected_by_children: Optional[List[PydanticObjectId]] = []
    # written strategy trial relevant field
    written_strategy: Optional[WrittenStrategy] = None
    # post survey trial relevant field
    post_survey: Optional[PostSurvey] = None
    # redirect url with the confirmation code
    redirect_url: Optional[str] = None
    is_practice: Optional[bool] = False
    trial_title: Optional[str] = ""
    # relevant for the social learning loop to determine if the trial is the last in the example
    last_trial_for_current_example: Optional[bool] = False
    # block idx within social learning
    social_learning_block_idx: Optional[int] = 0
    # unique network idx within social learning block
    block_network_idx: Optional[int] = 0

    class Config:
        orm_mode = True


class TrialSaved(BaseModel):
    message: Optional[Literal["Trial saved"]] = "Trial saved"


class TrialError(BaseModel):
    message: Literal[
        "Trial type is not correct",
        "Trial results are missing",
        "Advisor session is not found",
    ]


class SessionError(BaseModel):
    message: str
