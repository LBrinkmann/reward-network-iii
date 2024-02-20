from typing import List

from fastapi import Depends, APIRouter, HTTPException
from fastapi.security import HTTPBasicCredentials
from fastapi.responses import HTMLResponse

from models.session import Session
from models.subject import Subject
from routes.security_utils import get_user

results_router = APIRouter(tags=["Results"])


@results_router.get('/sessions')
async def get_results(
        experiment_type: str = None,
        completed: bool = None,
        user: HTTPBasicCredentials = Depends(get_user)) -> List[Session]:
    search_criteria = []
    if experiment_type is not None:
        search_criteria.append({'experiment_type': experiment_type})
    if completed is not None:
        search_criteria.append({'completed': completed})

    if len(search_criteria) == 0:
        sessions = await Session.find().to_list()
    else:
        sessions = await Session.find({"$and": search_criteria}).to_list()
    # return html document with the progress graph
    return sessions


@results_router.get('/subjects')
async def get_results(
        user: HTTPBasicCredentials = Depends(get_user)) -> List[Subject]:
    subjects = await Subject.find().to_list()
    # return html document with the progress graph
    return subjects


@results_router.get('/statistic/{experiment_type}', response_class=HTMLResponse)
async def get_statistic(
        experiment_type: str = None,
        user: HTTPBasicCredentials = Depends(get_user)) -> str:
    statistic = await Session.find(Session.experiment_type == experiment_type).aggregate([
        {
            "$group": {
                "_id": None, # Sums across all documents; remove or change for individual documents
                "total count": { "$sum": 1 }, # Counts all documents
                "available count": { "$sum": { "$cond": ["$available", 1, 0] } },
                "completed count": { "$sum": { "$cond": ["$completed", 1, 0] } },
                "replaced count": { "$sum": { "$cond": ["$replaced", 1, 0] } },
                "expired count": { "$sum": { "$cond": ["$expired", 1, 0] } },
                "finished count": { "$sum": { "$cond": ["$finished", 1, 0] } },
                "active count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$started_at", None]},
                                    {"$ne": ["$finished", True]},
                                    {"$ne": ["$expired", True]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            }
        }
    ]).to_list(1)

    if not statistic:
        raise HTTPException(status_code=404, detail="No data found")

    formatted_string = "<pre>\n" + "\n".join(
        f"{key}: {value}" for key, value in statistic[0].items() if key != "_id"
    ) + "\n</pre>"

    return formatted_string
