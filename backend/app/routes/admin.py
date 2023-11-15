from datetime import datetime

from fastapi import Depends, APIRouter
from fastapi.security import HTTPBasicCredentials
from fastapi import Query

from models.config import ExperimentSettings
from routes.security_utils import get_user
from study_setup.generate_sessions import generate_experiment_sessions

admin_router = APIRouter(tags=["Admin"])


@admin_router.get("/config")
async def get_config(experiment_type: str = Query(None), user: HTTPBasicCredentials = Depends(get_user)):
    config = await ExperimentSettings.find_one(
        ExperimentSettings.experiment_type == experiment_type)

    # return config in json format
    return config.dict()


@admin_router.post("/config")
async def update_config(new_config: ExperimentSettings,
                        user: HTTPBasicCredentials = Depends(get_user)):
    config = await ExperimentSettings.find_one(
        ExperimentSettings.experiment_type == new_config.experiment_type)

    # check the experiment type
    if config is not None and new_config.rewrite_previous_data == False:
        return {
            "error": "Experiment type was not changed and rewrite_previous_data"
                     " is False."}

    # update config and make it inactive

    # create a new config
    new_config.created_at = datetime.now()
    await new_config.save()

    # generate sessions
    await generate_experiment_sessions(new_config)

    return new_config.dict()
